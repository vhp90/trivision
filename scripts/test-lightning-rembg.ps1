param(
  [Parameter(Mandatory = $true)]
  [string]$ImagePath,

  [string]$ApiBaseUrl = $env:LIGHTNING_TRELLIS_API_URL
)

Add-Type -AssemblyName System.Net.Http

if (-not (Test-Path -LiteralPath $ImagePath)) {
  throw "Image not found: $ImagePath"
}

if ([string]::IsNullOrWhiteSpace($ApiBaseUrl)) {
  throw "LIGHTNING_TRELLIS_API_URL is not configured."
}

$ApiBaseUrl = $ApiBaseUrl.TrimEnd('/')
$endpoint = "$ApiBaseUrl/rembg"
$fileName = [System.IO.Path]::GetFileName($ImagePath)
$fileStream = [System.IO.File]::OpenRead($ImagePath)

try {
  $content = New-Object System.Net.Http.MultipartFormDataContent
  $fileContent = New-Object System.Net.Http.StreamContent($fileStream)
  $fileContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("image/jpeg")
  $content.Add($fileContent, "image", $fileName)

  $client = New-Object System.Net.Http.HttpClient
  $response = $client.PostAsync($endpoint, $content).GetAwaiter().GetResult()
  $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()

  Write-Host "Status:" ([int]$response.StatusCode) $response.ReasonPhrase
  Write-Host "Endpoint:" $endpoint
  Write-Host "Response:"
  Write-Output $body
} finally {
  $fileStream.Dispose()
}
