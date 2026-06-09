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

$outputCsv = Join-Path $dataDir "load_history.csv"

$pythonScript = @'
import sys, os, shutil, tempfile
import pandas as pd

excel_path = sys.argv[1]
output_path = sys.argv[2]

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
required = {"Date", "Hr", "Total Meters plus Gens"}
missing = required - set(df.columns)
if missing:
    print(f"ERROR: Missing columns: {missing}", file=sys.stderr)
    sys.exit(1)

df = df.dropna(subset=list(required))
df["date"] = (
    df["Date"].astype(int).astype(str).str.zfill(6)
    .pipe(lambda s: pd.to_datetime(s, format="%y%m%d"))
    .dt.strftime("%Y-%m-%d")
)
df["hr"] = df["Hr"].astype(int)
df["load"] = pd.to_numeric(df["Total Meters plus Gens"], errors="coerce")
df = df[(df["hr"] >= 1) & (df["hr"] <= 24) & (df["load"] > 0)].dropna(subset=["load"])

df[["date", "hr", "load"]].to_csv(output_path, index=False)
print(f"Wrote {len(df)} rows to {output_path}")
'@

$tmpScript = [System.IO.Path]::GetTempFileName() + ".py"
try {
    Set-Content -Path $tmpScript -Value $pythonScript -Encoding utf8
    python $tmpScript $excelPath $outputCsv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Python export failed."
        exit 1
    }
} finally {
    if (Test-Path $tmpScript) { Remove-Item $tmpScript }
}

Write-Host "Data exported successfully to $outputCsv"

# Git operations
$today = Get-Date -Format "yyyy-MM-dd"
Push-Location $PSScriptRoot
try {
    git add data/load_history.csv
    if ($LASTEXITCODE -ne 0) { throw "git add failed." }

    git commit -m "Update load data $today"
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
