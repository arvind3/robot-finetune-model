$ErrorActionPreference = "Stop"

if (Test-Path .env) {
  Get-Content .env | ForEach-Object {
    if ($_ -match '^(HF_TOKEN|GH_TOKEN)=(.*)$') {
      $name = $Matches[1]
      $value = $Matches[2]
      if (-not [string]::IsNullOrWhiteSpace($value)) {
        Set-Item -Path "Env:$name" -Value $value
      }
    }
  }
}

if (!(Test-Path .venv)) {
  $venvCreated = $false
  $bootstrapCommands = @(
    @{ cmd = "python"; args = @("-m", "venv", ".venv") },
    @{ cmd = "python.exe"; args = @("-m", "venv", ".venv") },
    @{ cmd = "py"; args = @("-3", "-m", "venv", ".venv") },
    @{ cmd = "py"; args = @("-m", "venv", ".venv") }
  )

  foreach ($candidate in $bootstrapCommands) {
    try {
      & $candidate.cmd @($candidate.args)
      if ($LASTEXITCODE -eq 0 -and (Test-Path ".\.venv\Scripts\python.exe")) {
        $venvCreated = $true
        break
      }
    } catch {
      continue
    }
  }

  if (-not $venvCreated) {
    throw "Unable to create .venv. Install Python 3 on Windows and rerun this script."
  }
}

. .\.venv\Scripts\Activate.ps1

$pythonCmd = ".\.venv\Scripts\python.exe"
if (!(Test-Path $pythonCmd)) {
  throw "Virtual environment Python not found at .\.venv\Scripts\python.exe. Delete .venv and rerun."
}

& $pythonCmd -m pip install --upgrade pip
& $pythonCmd -m pip install -r requirements.txt

$repoUrl = ""
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if ($gitCmd) {
  $repoUrl = git config --get remote.origin.url
  if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    & $pythonCmd tools\github_setup.py
    $repoUrl = git config --get remote.origin.url
  }
} else {
  Write-Host "git not found in PowerShell PATH; using config github_repo_url fallback."
}

& $pythonCmd tools\build_dataset.py
& $pythonCmd tools\validate_dataset.py
& $pythonCmd tools\validate_eval_suite.py --enforce-balanced-categories

if ($env:HF_TOKEN) {
  & $pythonCmd tools\publish_dataset.py
} else {
  Write-Host "HF_TOKEN not set; skipping dataset publish"
}

$colabArgs = @("tools\prepare_colab.py")
if (-not [string]::IsNullOrWhiteSpace($repoUrl)) {
  $colabArgs += @("--repo-url", $repoUrl)
}
$colabUrl = & $pythonCmd @colabArgs
if (-not [string]::IsNullOrWhiteSpace($colabUrl)) {
  $autorunEnabled = "$env:COLAB_AUTORUN".ToLower() -in @("1", "true", "yes", "on")
  if ($autorunEnabled) {
    Write-Host "COLAB_AUTORUN enabled; attempting browser automation."
    $skipSetup = "$env:COLAB_AUTORUN_SKIP_SETUP".ToLower() -in @("1", "true", "yes", "on")
    if (-not $skipSetup) {
      & $pythonCmd -m pip install -r requirements-automation.txt
      & $pythonCmd -m playwright install chromium
    }

    $autorunArgs = @(
      "tools\autorun_colab_playwright.py",
      "--colab-url", $colabUrl,
      "--user-data-dir", $(if ($env:COLAB_USER_DATA_DIR) { $env:COLAB_USER_DATA_DIR } else { ".colab_playwright_profile" }),
      "--timeout-minutes", $(if ($env:COLAB_AUTORUN_TIMEOUT_MINUTES) { $env:COLAB_AUTORUN_TIMEOUT_MINUTES } else { "240" }),
      "--poll-seconds", $(if ($env:COLAB_AUTORUN_POLL_SECONDS) { $env:COLAB_AUTORUN_POLL_SECONDS } else { "30" }),
      "--download-dir", $(if ($env:COLAB_AUTORUN_DOWNLOAD_DIR) { $env:COLAB_AUTORUN_DOWNLOAD_DIR } else { "artifacts/colab_run" })
    )

    if ("$env:COLAB_AUTORUN_HEADLESS".ToLower() -in @("1", "true", "yes", "on")) {
      $autorunArgs += "--headless"
    }
    if ("$env:COLAB_AUTORUN_START_ONLY".ToLower() -in @("1", "true", "yes", "on")) {
      $autorunArgs += "--start-only"
    }

    & $pythonCmd @autorunArgs
  } else {
    Write-Host $colabUrl
    Write-Host "Open the URL above and click Run all"
    Write-Host "Tip: set COLAB_AUTORUN=1 to trigger Playwright automation."
  }
}
