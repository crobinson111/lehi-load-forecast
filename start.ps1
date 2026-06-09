# =============================================================================
# Lehi Power Load Forecast — Start App
# Run this each time you want to use the forecast tool.
# The app will be available at http://localhost:8000 in your browser.
# Press Ctrl+C in this window to stop it.
# =============================================================================

Set-Location $PSScriptRoot

if (-not (Test-Path "$PSScriptRoot\.env")) {
    Write-Host "ERROR: .env file not found. Run setup.ps1 first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Lehi Power Load Forecast" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Starting server..." -ForegroundColor Yellow
Write-Host "  When you see 'Uvicorn running', open your browser to:" -ForegroundColor Yellow
Write-Host ""
Write-Host "      http://localhost:8000" -ForegroundColor Green
Write-Host ""
Write-Host "  The model trains automatically on startup (~60-90 seconds)." -ForegroundColor Gray
Write-Host "  Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

python -m uvicorn main:app --port 8000
