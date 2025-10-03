#!/usr/bin/env python3
import subprocess, tempfile, pathlib, statistics, re

# ======= KONFIG =======
COMMIT = "HEAD"                 # Regressions-Commit (SHA oder "HEAD")
BUILD_CMD = "rm -rf build/ && export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) && pip install -v .".split(' ')       # Anpassen!
TEST_CMD  = ["python", "performance_tester.py"]        # Muss eine Zahl ausgeben
METRIC_DIRECTION = "higher_is_better"  # oder "higher_is_better"
ROUNDS = 1                           # Läufe pro Messung (Median)
IMPROVEMENT_PCT = 5.0                # ab x% gilt als "spürbar besser"
FILE_FILTER_REGEX = None             # z.B. r"\.(cc|cu|py)$" um einzuschränken
CLEAN_BEFORE_BUILD = True            # git clean -fdx vor jedem Build
# =======================

def run(cmd, cwd, quiet=False):
    res = subprocess.run(cmd, cwd=cwd, check=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not quiet:
        print(res.stdout, end="")
    return res.stdout

def parse_metric(out: str) -> float:
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", out)
    if not nums:
        raise RuntimeError("Benchmark-Ausgabe enthält keine Zahl.")
    return float(nums[-1])  # letzte Zahl nehmen

def measure(cwd: pathlib.Path) -> float:
    if CLEAN_BEFORE_BUILD:
        run(["bash", "-lc", "git clean -fdx"], cwd, quiet=True)
    run(BUILD_CMD, cwd)
    vals = []
    for _ in range(ROUNDS):
        out = run(TEST_CMD, cwd)
        vals.append(parse_metric(out))
    med = statistics.median(vals)
    print(f"[Benchmark] Werte={vals}  Median={med}")
    return med

def improvement_pct(new: float, base: float) -> float:
    if METRIC_DIRECTION == "lower_is_better":
        return (base - new) / base * 100.0
    else:
        return (new - base) / base * 100.0

def list_changed_files(repo: pathlib.Path, commit: str) -> list[str]:
    parent = f"{commit}^"
    out = run(["git","diff","--name-only","--diff-filter=AMR", parent, commit], repo, quiet=True)
    files = [l.strip() for l in out.splitlines() if l.strip()]
    if FILE_FILTER_REGEX:
        files = [f for f in files if re.search(FILE_FILTER_REGEX, f)]
    return files

def main():
    repo = pathlib.Path(".").resolve()
    # Sicherheit: sind wir in einem Git-Repo?
    subprocess.run(["git","rev-parse","--is-inside-work-tree"], cwd=repo, check=True,
                   stdout=subprocess.DEVNULL)

    parent = f"{COMMIT}^"
    files = list_changed_files(repo, COMMIT)
    if not files:
        print("Keine geänderten Dateien im Commit gefunden.")
        return

    print(f"Gefundene geänderte Dateien ({len(files)}):")
    for f in files: print("  -", f)

    with tempfile.TemporaryDirectory(prefix="perfscan-") as td:
        work = pathlib.Path(td)
        # Abgetrennte Worktree auf dem Regressions-Commit
        run(["git","worktree","add","--detach",str(work), COMMIT], repo)
        try:
            print("\n== Basis-Messung (alle Dateien neu) ==")
            baseline = measure(work)

            culprits = []
            current_base = baseline

            for f in files:
                print(f"\n>>> Probiere: {f} auf alten Stand ({parent}) zurücksetzen")
                # Nur diese Datei auf Parent zurücksetzen
                run(["git","restore","--source", parent, "--", f], work)

                trial = measure(work)
                imp = improvement_pct(trial, current_base)
                print(f"Ergebnis mit alter {f}: {trial:.6g}  (Δ={imp:+.2f} % vs. aktuelle Basis)")

                if imp >= IMPROVEMENT_PCT:
                    print(f"✔ {f} ist verdächtig → alte Version BEHALTEN.")
                    culprits.append(f)
                    # Neue Basis ist der bessere Zustand (falls mehrere Files beitragen)
                    current_base = trial
                else:
                    # neue Version wiederherstellen
                    run(["git","restore","--source", COMMIT, "--", f], work)
                    print("↩ Keine deutliche Verbesserung → neue Version wiederhergestellt.")

            print("\n=== Zusammenfassung ===")
            if culprits:
                for c in culprits:
                    print("* Schuldige Datei (mind. {:.1f}% Verbesserung): {}".format(IMPROVEMENT_PCT, c))
                print("\nArbeitsbaum mit den aktuell besten Kombinationen liegt hier:")
                print(" ", work)
                print("\nPatch gegen den Commit erzeugen (optional):")
                print(f"  git -C {work} diff {COMMIT} > /tmp/revert-fix.patch")
                print("…und anschließend gezielt fixen oder zurückportieren.")
            else:
                print("Keine einzelne Datei brachte ≥ {:.1f}% Verbesserung.".format(IMPROVEMENT_PCT))
                print("→ Nutze Gruppentests / Delta-Debugging (siehe Hinweis unten).")
        finally:
            # Worktree wegräumen (auskommentieren, falls du reinschauen willst)
            run(["git","worktree","remove","--force",str(work)], repo, quiet=True)

if __name__ == "__main__":
    main()
