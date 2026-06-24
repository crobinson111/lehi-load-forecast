# push-data.ps1 — export Excel data to CSV and push to GitHub

# Read EXCEL_PATH from .env
$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) {
    Write-Error ".env file not found at $envFile"
    exit 1
}

$excelPath = $null
Get-Content $envFile | ForEach-Object {
    if ($_ -match "^EXCEL_PATH=(.+)$") {
        $excelPath = $matches[1].Trim()
    }
}

if (-not $excelPath) {
    Write-Error "EXCEL_PATH not set in .env"
    exit 1
}

if (-not (Test-Path $excelPath)) {
    Write-Error "Excel file not found: $excelPath"
    exit 1
}

Write-Host "Reading Excel data from: $excelPath"

# Ensure data/ directory exists
$dataDir = Join-Path $PSScriptRoot "data"
if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
}

$outputCsv      = Join-Path $dataDir "load_history.csv"
$outputSolarCsv = Join-Path $dataDir "red_mesa_history.csv"

$pythonScript = @'
import sys, os, shutil, tempfile
import pandas as pd

excel_path       = sys.argv[1]
output_load      = sys.argv[2]
output_solar     = sys.argv[3]

try:
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    tmp.close()
    shutil.copy2(excel_path, tmp.name)
except Exception as exc:
    print(f"ERROR: Could not copy spreadsheet: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    df = pd.read_excel(tmp.name, sheet_name="Sch Log Data (2)", engine="openpyxl")
finally:
    os.unlink(tmp.name)

df.columns = df.columns.str.strip()

# --- Load history ---
required = {"Date", "Hr", "Total Meters plus Gens"}
missing = required - set(df.columns)
if missing:
    print(f"ERROR: Missing columns: {missing}", file=sys.stderr)
    sys.exit(1)

df_load = df.dropna(subset=list(required)).copy()
df_load["date"] = (
    df_load["Date"].astype(int).astype(str).str.zfill(6)
    .pipe(lambda s: pd.to_datetime(s, format="%y%m%d"))
    .dt.strftime("%Y-%m-%d")
)
df_load["hr"]   = df_load["Hr"].astype(int)
df_load["load"] = pd.to_numeric(df_load["Total Meters plus Gens"], errors="coerce")
df_load = df_load[(df_load["hr"] >= 1) & (df_load["hr"] <= 24) & (df_load["load"] > 0)].dropna(subset=["load"])
df_load[["date", "hr", "load"]].to_csv(output_load, index=False)
print(f"Wrote {len(df_load)} load rows to {output_load}")

# --- Red Mesa solar history ---
if "RED MESA" not in df.columns:
    print("WARNING: RED MESA column not found — skipping solar export", file=sys.stderr)
    sys.exit(0)

df_solar = df.dropna(subset=["Date", "Hr"]).copy()
df_solar["date"] = (
    df_solar["Date"].astype(int).astype(str).str.zfill(6)
    .pipe(lambda s: pd.to_datetime(s, format="%y%m%d"))
    .dt.strftime("%Y-%m-%d")
)
df_solar["hr"]  = df_solar["Hr"].astype(int)
df_solar["kwh"] = pd.to_numeric(df_solar["RED MESA"], errors="coerce").fillna(0)
df_solar = df_solar[(df_solar["hr"] >= 1) & (df_solar["hr"] <= 24)]
df_solar[["date", "hr", "kwh"]].to_csv(output_solar, index=False)
print(f"Wrote {len(df_solar)} solar rows to {output_solar}")
'@

$tmpScript = [System.IO.Path]::GetTempFileName() + ".py"
try {
    Set-Content -Path $tmpScript -Value $pythonScript -Encoding utf8
    python $tmpScript $excelPath $outputCsv $outputSolarCsv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Python export failed."
        exit 1
    }
} finally {
    if (Test-Path $tmpScript) { Remove-Item $tmpScript }
}

Write-Host "Data exported successfully."

# Git operations
$today = Get-Date -Format "yyyy-MM-dd"
Push-Location $PSScriptRoot
try {
    git add data/load_history.csv data/red_mesa_history.csv
    if ($LASTEXITCODE -ne 0) { throw "git add failed." }

    git commit -m "Update load + Red Mesa data $today"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed." }

    git push
    if ($LASTEXITCODE -ne 0) { throw "git push failed." }

    Write-Host "Successfully pushed data update for $today."
} catch {
    Write-Error $_.Exception.Message
    exit 1
} finally {
    Pop-Location
}
