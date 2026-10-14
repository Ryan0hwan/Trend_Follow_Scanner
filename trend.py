"""
미너비니 추세 템플릿 스캐너
- S&P 500 + 내 관심종목에서 마크 미너비니 추세 템플릿 8개 조건을 매일 확인해
- ① 8개를 모두 만족하는 종목 전체
- ② 그중 최근 10거래일 안에 새로 템플릿에 들어온 종목 (B, 과열 제외)
  을 디스코드로 따로 보냅니다.
"""
import datetime as dt
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ───────────── 설정 (여기만 고치면 됩니다) ─────────────
GROUPS = os.environ.get("SCAN_GROUPS", "sp500,watchlist").split(",")

RS_MIN = float(os.environ.get("RS_MIN", "70"))                  # 8. RS 하한
LOW_UP_PCT = float(os.environ.get("LOW_UP_PCT", "30"))          # 6. 52주 저점 대비 최소 상승률(%)
HIGH_DOWN_PCT = float(os.environ.get("HIGH_DOWN_PCT", "25"))    # 7. 52주 고점 대비 최대 하락률(%)
SLOPE_DAYS = int(os.environ.get("SLOPE_DAYS", "21"))            # 3. 200일선 상승 확인 기간(거래일, 약 1개월)

NEW_DAYS = int(os.environ.get("NEW_DAYS", "10"))                # B. 템플릿에 들어온 지 이 거래일 이내
MAX_ABOVE_50 = float(os.environ.get("MAX_ABOVE_50", "25"))      # B. 과열 제외: 50일선 위 이 % 초과면 제외

MIN_PRICE = float(os.environ.get("MIN_PRICE", "5"))
MIN_DOLLAR_VOL = float(os.environ.get("MIN_DOLLAR_VOL", "5000000"))  # 50일 평균 거래대금($)

WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE = os.environ.get("FORCE", "0") == "1"                     # 주말에도 실행 (최근 거래일 기준)
# ──────────────────────────────────────────────────────

NY = ZoneInfo("America/New_York")
STATE = Path(__file__).parent / "state"
STATE.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (trend-scanner)"}
GROUP_LABEL = {"sp500": "S&P 500", "watchlist": "Watchlist"}
HIST = NEW_DAYS + 1                     # 오늘 포함 최근 며칠의 조건 충족 여부를 계산할지


# ───────────── 종목 목록 ─────────────
def _fetch_sp500() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers=UA, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    return pd.DataFrame({"ticker": t["Symbol"], "name": t["Security"], "industry": t["GICS Sub-Industry"]})


def _load_watchlist() -> pd.DataFrame:
    p = Path(__file__).parent / "watchlist.txt"
    rows = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if not ln:
                continue
            tk, _, name = ln.partition(",")
            tk = re.sub(r"\.([A-Z])$", r"-\1", tk.strip().upper())  # MOG.A → MOG-A
            rows.append({"ticker": tk, "name": name.strip(), "industry": "Watchlist"})
    return pd.DataFrame(rows, columns=["ticker", "name", "industry"]).drop_duplicates("ticker")


def load_universe(group: str) -> pd.DataFrame:
    if group == "watchlist":
        return _load_watchlist()
    cache = STATE / f"universe_{group}.csv"
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < 7 * 86400
    if not fresh:
        try:
            df = {"sp500": _fetch_sp500}[group]()
            if len(df) < 50:
                raise ValueError("종목 수가 비정상적으로 적음")
            df.to_csv(cache, index=False)
        except Exception as e:
            print(f"[경고] {group} 목록 갱신 실패: {e}")
    if not cache.exists():
        return pd.DataFrame(columns=["ticker", "name", "industry"])
    df = pd.read_csv(cache).fillna("")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
    df = df[df["ticker"].str.fullmatch(r"[A-Z\-]{1,6}")]
    return df.drop_duplicates("ticker")


# ───────────── 시세 ─────────────
def download(tickers: list[str]):
    import yfinance as yf
    for i in range(0, len(tickers), 200):
        chunk = tickers[i:i + 200]
        try:
            data = yf.download(chunk, period="2y", interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print(f"[경고] 시세 다운로드 실패({chunk[0]}…): {e}")
            continue
        if not isinstance(data.columns, pd.MultiIndex):
            data = pd.concat({chunk[0]: data}, axis=1)
        have = set(data.columns.get_level_values(0))
        for t in chunk:
            if t in have:
                yield t, data[t]
        time.sleep(1)


# ───────────── 조건 계산 ─────────────
def rs_score(c: pd.Series) -> pd.Series:
    """IBD 방식 가중 수익률: 3개월 40% + 6·9·12개월 각 20% (기간이 모자라면 있는 기간만으로)"""
    parts = [(63, 2), (126, 1), (189, 1), (252, 1)]
    num = sum((c / c.shift(n) - 1).fillna(0) * w for n, w in parts)
    den = sum(c.shift(n).notna() * w for n, w in parts)
    return num / den.replace(0, np.nan)


def price_checks(df: pd.DataFrame) -> pd.DataFrame | None:
    """최근 HIST일 각각에 대해 조건 1~7 충족 여부와 표시용 값 (조건 8 RS는 전체 종목 비교 후 따로)"""
    df = df.dropna(subset=["Close", "High", "Low", "Volume"])
    if len(df) < 200 + SLOPE_DAYS + HIST:
        return None
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    if c.iloc[-1] < MIN_PRICE or (c * v).tail(50).mean() < MIN_DOLLAR_VOL:
        return None
    s50, s150, s200 = c.rolling(50).mean(), c.rolling(150).mean(), c.rolling(200).mean()
    hi52, lo52 = h.rolling(252, min_periods=200).max(), l.rolling(252, min_periods=200).min()
    out = pd.DataFrame({
        "c1": (c > s150) & (c > s200),
        "c2": s150 > s200,
        "c3": s200 > s200.shift(SLOPE_DAYS),
        "c4": (s50 > s150) & (s50 > s200),
        "c5": c > s50,
        "c6": c >= lo52 * (1 + LOW_UP_PCT / 100),
        "c7": c >= hi52 * (1 - HIGH_DOWN_PCT / 100),
        "score": rs_score(c),
        "close": c,
        "above50": (c / s50 - 1) * 100,
        "from_hi": (c / hi52 - 1) * 100,
        "from_lo": (c / lo52 - 1) * 100,
        "r1m": (c / c.shift(21) - 1) * 100,
        "r3m": (c / c.shift(63) - 1) * 100,
    }).tail(HIST)
    out["date"] = [d.date() for d in out.index]
    return out.reset_index(drop=True)


def evaluate(per_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """RS 백분위를 날짜별로 계산하고, 8개 조건 충족 여부와 템플릿 진입 후 일수를 붙임"""
    # 날짜를 맞춰 쌓기 (종목마다 마지막 거래일이 같다고 보고 뒤에서부터 정렬)
    frames = []
    for t, f in per_ticker.items():
        g = f.copy()
        g["t"] = t
        g["k"] = range(len(g) - HIST, len(g))       # 0 = 가장 오래된 날, HIST-1 = 오늘
        frames.append(g)
    allf = pd.concat(frames, ignore_index=True)
    allf["rs"] = allf.groupby("k")["score"].rank(pct=True) * 98 + 1
    allf["c8"] = allf["rs"] >= RS_MIN
    conds = [f"c{i}" for i in range(1, 9)]
    allf["all8"] = allf[conds].all(axis=1)

    rows = []
    for t, g in allf.groupby("t"):
        g = g.sort_values("k")
        today = g.iloc[-1]
        if not today["all8"]:
            continue
        # 오늘부터 거꾸로 연속으로 8개를 만족한 일수
        streak = 0
        for ok in reversed(g["all8"].tolist()):
            if not ok:
                break
            streak += 1
        rows.append({
            "t": t, "rs": int(round(today["rs"])), "days": streak if streak < HIST else None,
            "close": float(today["close"]), "above50": float(today["above50"]),
            "from_hi": float(today["from_hi"]), "from_lo": float(today["from_lo"]),
            "r1m": float(today["r1m"]), "r3m": float(today["r3m"]),
            "date": str(today["date"]),
        })
    return pd.DataFrame(rows)


# ───────────── 디스코드 ─────────────
def company_name(ticker: str) -> str:
    cache_file = STATE / "names.json"
    try:
        cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    except Exception:
        cache = {}
    if ticker in cache:
        return cache[ticker]
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ""
    except Exception:
        name = ""
    cache[ticker] = name
    cache_file.write_text(json.dumps(cache))
    return name


def post(content: str):
    if not WEBHOOK:
        print(content)
        return
    for _ in range(5):
        r = requests.post(WEBHOOK, json={"content": content}, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 2)) + 0.5)
            continue
        r.raise_for_status()
        time.sleep(0.6)
        return


def send_table(title: str, rows: list[dict], with_name: bool):
    nm = f"{'NAME':<16}" if with_name else ""
    head = (f"{'TICKER':<7}{nm}{'RS':>3}{'DAYS':>5}{'LAST':>9}{'50D%':>7}{'HI%':>7}{'LO%':>7}{'1M%':>7}{'3M%':>7}")
    def line(r):
        d = r["days"]
        days = f"{NEW_DAYS}+" if d is None or (isinstance(d, float) and np.isnan(d)) else str(int(d))
        name = f"{r['name'][:14]:<16}" if with_name else ""
        return (f"{r['t']:<7}{name}{r['rs']:>3}{days:>5}{r['close']:>9.2f}{r['above50']:>+7.1f}"
                f"{r['from_hi']:>+7.1f}{r['from_lo']:>+7.0f}{r['r1m']:>+7.1f}{r['r3m']:>+7.1f}")
    lines = [line(r) for r in rows]
    first, part = True, 1
    while lines:
        block, size = [], 0
        while lines and size + len(lines[0]) + 1 < 1750:
            size += len(lines[0]) + 1
            block.append(lines.pop(0))
        header = f"**{title}**\n" if first else f"**{title} ({part})**\n"
        post(f"{header}```\n{head}\n" + "\n".join(block) + "\n```")
        first, part = False, part + 1


# ───────────── 실행 ─────────────
def main():
    now = dt.datetime.now(NY)
    if now.weekday() >= 5 and not FORCE:
        print("주말. 종료.")
        return

    universes = {g.strip(): load_universe(g.strip()) for g in GROUPS if g.strip()}
    meta, group_of = {}, {}
    for g, u in universes.items():
        for row in u.itertuples():
            if row.ticker not in meta:
                meta[row.ticker] = row
                group_of[row.ticker] = g
    print(f"스캔 대상 {len(meta)}종목 " + ", ".join(f"{g}={len(u)}" for g, u in universes.items()))
    if not meta:
        post("⚠️ 추세 템플릿 스캐너: 종목 목록을 불러오지 못했습니다. Actions 로그를 확인하세요.")
        return

    per_ticker, scanned = {}, 0
    for ticker, df in download(list(meta)):
        scanned += 1
        f = price_checks(df)
        if f is not None:
            per_ticker[ticker] = f
    if not per_ticker:
        post("⚠️ 추세 템플릿 스캐너: 시세를 받지 못했습니다. 일시적 오류일 수 있습니다.")
        return
    # 마지막 거래일이 다른 종목(거래정지 등)은 제외
    asof = pd.Series([f["date"].iloc[-1] for f in per_ticker.values()]).mode().iloc[0]
    per_ticker = {t: f for t, f in per_ticker.items() if f["date"].iloc[-1] == asof}

    res = evaluate(per_ticker)
    for col in ("name", "group"):
        res[col] = ""
    for i, r in res.iterrows():
        m = meta[r["t"]]
        res.at[i, "name"] = str(m.name) or company_name(r["t"])
        res.at[i, "group"] = group_of[r["t"]]
    res = res.sort_values(["rs", "r3m"], ascending=False)

    # ② B: 최근 NEW_DAYS일 안에 새로 진입 + 과열 제외
    newb = res[res["days"].notna() & (res["above50"] <= MAX_ABOVE_50)].sort_values(["days", "rs"], ascending=[True, False])
    dstr = f"{asof:%b} {asof.day}"
    post(f"**{dstr} | 미너비니 추세 템플릿**\n"
         f"8개 조건 모두 충족 **{len(res)}**종목 · 그중 최근 {NEW_DAYS}거래일 신규 진입 **{len(newb)}**종목 (과열 제외)")
    if len(newb):
        send_table(f"② 신규 진입 (B) — {NEW_DAYS}거래일 이내, 50일선 +{MAX_ABOVE_50:.0f}% 이하", newb.to_dict("records"), True)
    else:
        post(f"② 신규 진입 (B): 오늘은 해당 종목이 없습니다.")

    # ① 8개 모두 충족 전체 (그룹별)
    for g in universes:
        rows = res[res["group"] == g].to_dict("records")
        if rows:
            send_table(f"① 8개 조건 모두 충족 — {GROUP_LABEL.get(g, g)} ({len(rows)}종목)", rows, False)

    if scanned < len(meta) * 0.5:
        post(f"⚠️ 추세 템플릿 스캐너: 시세를 {scanned}/{len(meta)}종목만 받았습니다. 일시적 오류일 수 있습니다.")
    print(f"완료: {asof}, 확인 {len(per_ticker)}종목, 충족 {len(res)}, 신규 {len(newb)}")


if __name__ == "__main__":
    sys.exit(main())
