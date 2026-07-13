# auto-push-data.ps1
# Checks whether the Excel spreadsheet has dates newer than the last pushed
# load_history.csv. Runs push-data.ps1 only when new data is available.
# Schedule daily via Windows Task Scheduler (see bottom of this file).

$scriptDir = $PSScriptRoot
$csvPath   = Join-Path $scriptDir "data\load_history.csv"

if (-not (Test-Path $csvPath)) {
    Write-Host "load_history.csv not found — running full push."
    powershell -File (Join-Path $scriptDir "push-data.ps1")
    exit $LASTEXITCODE
}

# Quick Python check: compare max date in Excel vs max date in CSV
$checkPy = @'
import os, sys, shutil, tempfile
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
csv_path   = sys.argv[1]
excel_path = os.environ.get("EXCEL_PATH", "")

df_csv  = pd.read_csv(csv_path)
csv_max = df_csv["date"].max()

tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
tmp.close()
shutil.copy2(excel_path, tmp.name)
try:
    df_xl = pd.read_excel(tmp.name, sheet_name="Sch Log Data (2)", engine="openpyxl")
finally:
    os.unlink(tmp.name)

df_xl.columns = df_xl.columns.str.strip()
df_xl = df_xl.dropna(subset=["Date", "Hr"])
df_xl["date"] = (
    pd.to_datetime(df_xl["Date"].astype(int).astype(str).str.zfill(6), format="%y%m%d")
    .dt.strftime("%Y-%m-%d")
)
xl_max = df_xl["date"].max()

if xl_max > csv_max:
    print("NEW:" + xl_max)
else:
    print("CURRENT:" + csv_max)
'@

$pyFile = Join-Path $env:TEMP "lehi_check_new_data.py"
[System.IO.File]::WriteAllText($pyFile, $checkPy, (New-Object System.Text.UTF8Encoding $false))

Push-Location $scriptDir
try {
    $result = python $pyFile $csvPath 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Check script failed: $result"
        exit 1
    }

    if ($result -like "NEW:*") {
        $newDate = $result -replace "NEW:", ""
        Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm') — New data through $newDate, pushing..."
        powershell -File (Join-Path $scriptDir "push-data.ps1")
        exit $LASTEXITCODE
    } else {
        $cur = $result -replace "CURRENT:", ""
        Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm') — Already current through $cur, nothing to push."
    }
} finally {
    Pop-Location
    Remove-Item $pyFile -ErrorAction SilentlyContinue
}
