# Build the DCT chunk-major CUDA kernel DLL for Windows
# Requires: MSVC (Visual Studio Build Tools) + CUDA Toolkit 12.x
param(
    [string]$Arch = "sm_86"  # RTX 3060/4060=sm_86, RTX 3090/4090=sm_89, A100=sm_80, H100=sm_90
)

$cu = Join-Path $PSScriptRoot "dct_chunk_major_kernel.cu"
$dll = Join-Path $PSScriptRoot "dct_kernel_chunk_major.dll"

nvcc -shared -o "$dll" "$cu" -arch=$Arch -O2 --use_fast_math

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK -> $dll (arch=$Arch)"
} else {
    Write-Host "FAILED (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}
