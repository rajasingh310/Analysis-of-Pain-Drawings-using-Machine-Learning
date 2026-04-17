param(
    [Parameter(Mandatory = $true)]
    [string]$SvgPath,

    [Parameter(Mandatory = $true)]
    [string]$PdfPath,

    [Parameter(Mandatory = $true)]
    [string]$EdgePath,

    [Parameter(Mandatory = $true)]
    [string]$UserDataDir
)

$svgContent = Get-Content -LiteralPath $SvgPath -Raw -Encoding UTF8

$widthMatch = [regex]::Match($svgContent, 'width="([0-9.]+)pt"')
$heightMatch = [regex]::Match($svgContent, 'height="([0-9.]+)pt"')

if (-not $widthMatch.Success -or -not $heightMatch.Success) {
    throw "Could not read width/height in pt from SVG: $SvgPath"
}

$widthPt = $widthMatch.Groups[1].Value
$heightPt = $heightMatch.Groups[1].Value

$htmlPath = [System.IO.Path]::ChangeExtension($PdfPath, ".html")

$html = @"
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    @page {
      size: ${widthPt}pt ${heightPt}pt;
      margin: 0;
    }
    html, body {
      margin: 0;
      padding: 0;
      width: ${widthPt}pt;
      height: ${heightPt}pt;
      overflow: hidden;
      background: white;
    }
    svg {
      display: block;
      width: ${widthPt}pt;
      height: ${heightPt}pt;
    }
  </style>
</head>
<body>
$svgContent
</body>
</html>
"@

Set-Content -LiteralPath $htmlPath -Value $html -Encoding UTF8

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $PdfPath) | Out-Null
Remove-Item -LiteralPath $UserDataDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $UserDataDir | Out-Null

$fileUrl = "file:///" + ($htmlPath -replace "\\", "/")

Start-Process -FilePath $EdgePath -ArgumentList @(
    '--headless',
    '--disable-gpu',
    '--no-pdf-header-footer',
    "--user-data-dir=$UserDataDir",
    "--print-to-pdf=$PdfPath",
    $fileUrl
) -Wait -NoNewWindow
