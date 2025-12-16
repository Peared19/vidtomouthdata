import urllib.request, json, sys

url = 'http://localhost:8000/animate'
text = ("The quick brown fox jumps over the lazy dog while the mysterious wind whispers "
        "through the ancient forest, carrying tales of forgotten civilizations and lost treasures "
        "hidden beneath the moonlit sky.")

data = json.dumps({'text': text}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = resp.read()
        print('STATUS', resp.status)
        print('LENGTH', len(body))
        # try pretty print start
        try:
            j = json.loads(body)
            print('keys:', list(j.keys()))
            if 'frames' in j:
                print('frames:', len(j['frames']))
            if 'audio_url' in j:
                print('audio_url:', j['audio_url'])
        except Exception as e:
            print('Response not JSON or parse failed:', e)
            print(body[:1000])
except Exception as e:
    print('Request failed:', e)
    sys.exit(1)
