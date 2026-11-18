"""Download CCTV-Fights (Original Format) from ROSE Lab directly to the server.

Run it yourself in a terminal; it asks for your ROSE username/password (password hidden).
Credentials are only sent to rose1.ntu.edu.sg and are never printed or saved.
Usage:  python download_cctv_fights.py            # the five Original Format zips (ids 210-214)
        python download_cctv_fights.py 215 216     # other file ids from the download page
"""
import getpass, re, sys, zipfile
from pathlib import Path
import requests

BASE = "https://rose1.ntu.edu.sg"
PAGE = f"{BASE}/dataset/cctvFights/download"
OUT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/cctvfights")
IDS = [int(a) for a in sys.argv[1:]] or [210, 211, 212, 213, 214]


def login(s):
    url = f"{BASE}/login/"
    r = s.get(url, timeout=30); r.raise_for_status()
    tok = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text).group(1)
    user = input("ROSE username: ").strip()
    pw = getpass.getpass("ROSE password (hidden): ")
    r = s.post(url, data={"csrfmiddlewaretoken": tok, "username": user, "password": pw},
               headers={"Referer": url}, timeout=30)
    del pw
    if "/login" in r.url or 'name="password"' in r.text:
        sys.exit("Login failed: check username/password.")
    print("Logged in.")


def list_links(s):
    """Print every file offered on the download page (to spot the annotation file)."""
    html = s.get(PAGE, timeout=30).text
    links = re.findall(r'href="(/dataset/cctvFights/download/(\d+))"[^>]*>\s*([^<]+?)\s*<', html)
    (OUT / "download_page_links.txt").write_text("\n".join(f"{i}\t{n}" for _, i, n in links) + "\n")
    print(f"{len(links)} files on the page (list saved to download_page_links.txt):")
    for _, i, n in links:
        print(f"  {i}  {n}")


def fetch(s, fid):
    with s.get(f"{PAGE}/{fid}", stream=True, timeout=60) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            print(f"  id {fid}: got an HTML page, not a file (skipped)"); return
        cd = r.headers.get("Content-Disposition", "")
        m = re.search(r'filename="?([^";]+)"?', cd)
        name = m.group(1) if m else f"cctvfights_{fid}.bin"
        dst = OUT / name
        total = int(r.headers.get("Content-Length", 0))
        if dst.exists() and total and dst.stat().st_size == total:
            print(f"  {name}: already complete"); return
        tmp = dst.with_suffix(dst.suffix + ".part")
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(8 << 20):
                f.write(chunk); done += len(chunk)
                if total:
                    print(f"\r  {name}: {done/2**30:.2f} / {total/2**30:.2f} GB", end="", flush=True)
        print()
        tmp.rename(dst)
    if dst.suffix == ".zip":
        bad = zipfile.ZipFile(dst).testzip()
        print(f"  {name}: {'OK' if bad is None else 'CORRUPT at ' + bad}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with requests.Session() as s:
        s.headers["User-Agent"] = "Mozilla/5.0"
        login(s)
        list_links(s)
        for fid in IDS:
            fetch(s, fid)
    print("DOWNLOAD FINISHED ->", OUT)
