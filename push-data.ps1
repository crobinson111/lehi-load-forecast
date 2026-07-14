# push-data.ps1 - export Excel data + training weather to CSV and push to GitHub

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

$fetchScript = Join-Path $PSScriptRoot "fetch_weather.py"
python $fetchScript $excelPath `
    (Join-Path $dataDir "load_history.csv") `
    (Join-Path $dataDir "red_mesa_history.csv") `
    (Join-Path $dataDir "load_weather.csv") `
    (Join-Path $dataDir "solar_weather.csv") `
    (Join-Path $dataDir "steele_a_history.csv") `
    (Join-Path $dataDir "steele_a_weather.csv") `
    (Join-Path $dataDir "supply_history.csv") `
    (Join-Path $dataDir "h_butte_weather.csv")

if ($LASTEXITCODE -ne 0) {
    Write-Error "Data export failed."
    exit 1
}

Write-Host "All data exported successfully."

# Fetch UAMPS schedule (CRSP, Provo River, Veyo) - 30 days back + 16 days forward
$uampsScript = Join-Path $PSScriptRoot "fetch_uamps.py"
$uampsOut    = Join-Path $dataDir "uamps_schedule.csv"
Write-Host "Fetching UAMPS scheduler data..."
python $uampsScript $uampsOut 30 16
if ($LASTEXITCODE -ne 0) {
    Write-Warning "UAMPS fetch failed - continuing without it."
}

# Git operations
$today = Get-Date -Format "yyyy-MM-dd"
Push-Location $PSScriptRoot
try {
    git add data/load_history.csv data/red_mesa_history.csv data/load_weather.csv `
            data/solar_weather.csv data/steele_a_history.csv data/steele_a_weather.csv `
            data/supply_history.csv data/h_butte_weather.csv data/uamps_schedule.csv
    if ($LASTEXITCODE -ne 0) { throw "git add failed." }

    git diff --staged --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m "Update load, solar, weather, and UAMPS schedule data $today"
        if ($LASTEXITCODE -ne 0) { throw "git commit failed." }
        git push
        if ($LASTEXITCODE -ne 0) { throw "git push failed." }
        Write-Host "Successfully pushed data update for $today."
    } else {
        Write-Host "No changes to push - data already up to date."
    }
} catch {
    Write-Error $_.Exception.Message
    exit 1
} finally {
    Pop-Location
}
