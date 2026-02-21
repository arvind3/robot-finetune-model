$ErrorActionPreference = "Stop"

if (Test-Path .env) {
  Get-Content .env | ForEach-Object {
    if ($_ -match '^(HF_TOKEN|GH_TOKEN)=(.*)$') {
      $name = $Matches[1]
      $value = $Matches[2]
      if (-not [string]::IsNullOrWhiteSpace($value)) {
        $env:$name = $value
      }
    }
  }
}

if (!(Test-Path .venv)) {
  python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

$repoUrl = git config --get remote.origin.url
if ([string]::IsNullOrWhiteSpace($repoUrl)) {
  python tools\github_setup.py
  $repoUrl = git config --get remote.origin.url
}

python tools\build_dataset.py
python tools\validate_dataset.py

if ($env:HF_TOKEN) {
  python tools\publish_dataset.py
} else {
  Write-Host "HF_TOKEN not set; skipping dataset publish"
}

$colabUrl = python tools\prepare_colab.py --repo-url $repoUrl
if (-not [string]::IsNullOrWhiteSpace($colabUrl)) {
  Write-Host $colabUrl
  Write-Host "Open the URL above and click Run all"
}