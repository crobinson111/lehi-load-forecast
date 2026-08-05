# push_realtime.ps1 - fetch today's UAMPS meter load and push to GitHub
# Schedule via Windows Task Scheduler: every hour at :15

$script = Join-Path $PSScriptRoot "fetch_realtime_load.py"
python $script
if ($LASTEXITCODE -ne 0) {
    Write-Error "fetch_realtime_load.py failed"
    exit 1
}

$today = Get-Date -Format "yyyy-MM-dd"
Push-Location $PSScriptRoot
try {
    git add data/realtime_load.json data/realtime_history.json
    git diff --staged --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m "Realtime load update $today"
        if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
        git push
        if ($LASTEXITCODE -ne 0) { throw "git push failed" }
        Write-Host "Pushed realtime load update for $today"
    } else {
        Write-Host "No changes to push"
    }
} catch {
    Write-Error $_.Exception.Message
    exit 1
} finally {
    Pop-Location
}
