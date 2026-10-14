# Render every slide of media\out\neural_ik_deck.pptx to PNG with PowerPoint (for checking layout).
#   powershell -ExecutionPolicy Bypass -File media\scripts\render_deck.ps1
param([string]$Deck = "$PSScriptRoot\..\out\neural_ik_deck.pptx", [string]$OutDir = "$PSScriptRoot\..\assets\deck_render")
$Deck = (Resolve-Path $Deck).Path
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
Get-ChildItem $OutDir -Filter "slide_*.png" | Remove-Item
$ppt = New-Object -ComObject PowerPoint.Application
try {
    $pres = $ppt.Presentations.Open($Deck, $true, $false, $false)   # read-only, no window
    foreach ($s in $pres.Slides) {
        $s.Export((Join-Path $OutDir ("slide_{0:D2}.png" -f $s.SlideIndex)), "PNG", 1920, 1080)
    }
    $n = $pres.Slides.Count
    $pres.Close()
    Write-Output "rendered $n slides to $OutDir"
} finally {
    $ppt.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($ppt) | Out-Null
}
