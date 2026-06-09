# =============================================================================
# Lehi Power Load Forecast — One-Time Setup
# Run this script once before using the app for the first time.
# =============================================================================

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Lehi Power Load Forecast — Setup" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# --- 1. Check Python ----------------------------------------------------------
Write-Host "Checking Python..." -ForegroundColor Yellow
try {
    $pyVersion = python --version 2>&1
    Write-Host "  Found: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found. Install Python 3.11 or later from https://www.python.org" -ForegroundColor Red
    Write-Host "  Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# --- 2. Install dependencies --------------------------------------------------
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow
Set-Location $PSScriptRoot
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: pip install failed. See output above." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "  Dependencies installed." -ForegroundColor Green

# --- 3. Configure Excel file path --------------------------------------------
Write-Host ""
Write-Host "Configure your data source" -ForegroundColor Yellow
Write-Host "  This is the full path to your scheduling log Excel file."
Write-Host "  Example: C:\Users\yourname\OneDrive - Lehi City\Scheduling - Documents\SchLogData.xlsx"
Write-Host ""
$excelPath = Read-Host "  Paste the full path to your Excel file"

if (-not (Test-Path $excelPath)) {
    Write-Host ""
    Write-Host "  WARNING: File not found at that path." -ForegroundColor Yellow
    Write-Host "  Double-check the path. The app will fail to train if the file is missing."
    Write-Host "  You can update it later by editing the .env file in this folder."
}

# Write .env file
"EXCEL_PATH=$excelPath" | Out-File -FilePath "$PSScriptRoot\.env" -Encoding utf8
Write-Host "  Saved to .env" -ForegroundColor Green

# --- 4. School calendar reminder ---------------------------------------------
Write-Host ""
Write-Host "School Calendar" -ForegroundColor Yellow
Write-Host "  Open school_calendar.csv in this folder and verify the school-break"
Write-Host "  dates are correct for Alpine School District. Update it each August"
Write-Host "  when the new school year calendar is published."

# --- Done --------------------------------------------------------------------
Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "  Run start.ps1 to launch the app." -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
