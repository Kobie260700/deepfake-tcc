$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BuildRoot = Join-Path $ProjectRoot ".detector-build"
$VenvRoot = Join-Path $BuildRoot "venv"
$Python = Join-Path $VenvRoot "Scripts\python.exe"
$WeightsRoot = Join-Path $BuildRoot "weights"
$Checkpoint = Join-Path $WeightsRoot "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36"
$Output = Join-Path $ProjectRoot "models\dfdc_efficientnet_b7.onnx"

if (-not (Test-Path $BuildRoot)) {
    New-Item -ItemType Directory -Path $BuildRoot | Out-Null
}

if (-not (Test-Path $Python)) {
    python -m venv $VenvRoot
}

$DependenciesReady = & $Python -c "import torch,timm,onnx; print('ok')" 2>$null
if ($DependenciesReady -ne "ok") {
    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Falha ao atualizar pip." }
    & $Python -m pip install "numpy<2" torch torchvision timm==0.6.13 onnx
    if ($LASTEXITCODE -ne 0) { throw "Falha ao instalar dependencias do detector." }
}

if (-not (Test-Path $Checkpoint)) {
    New-Item -ItemType Directory -Path $WeightsRoot -Force | Out-Null
    $Partial = "$Checkpoint.part"
    curl.exe -L --fail --progress-bar `
        -o $Partial `
        "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36"
    if ($LASTEXITCODE -ne 0) { throw "Falha ao baixar o checkpoint DFDC." }
    Move-Item -LiteralPath $Partial -Destination $Checkpoint
}

& $Python (Join-Path $PSScriptRoot "export_dfdc_to_onnx.py") `
    --checkpoint $Checkpoint `
    --output $Output
if ($LASTEXITCODE -ne 0) { throw "Falha ao exportar o detector para ONNX." }

Write-Host "Detector configurado em $Output"
