param(
    [string]$BiberTemp = 'C:\CodexData\tmp\biber'
)

$ErrorActionPreference = 'Stop'
$savedEnvironment = @{}
foreach ($name in @('TEMP', 'TMP', 'PAR_GLOBAL_TEMP', 'LC_ALL', 'LANG')) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

New-Item -ItemType Directory -Force -Path $BiberTemp | Out-Null
Push-Location $PSScriptRoot
try {
    $env:TEMP = $BiberTemp
    $env:TMP = $BiberTemp
    $env:PAR_GLOBAL_TEMP = $BiberTemp
    $env:LC_ALL = 'C'
    $env:LANG = 'C'
    & latexmk -g -xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "LaTeX build failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
}
