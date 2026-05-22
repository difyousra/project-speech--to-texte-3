#!/bin/sh
set -e
CERT="/app/certs/cert.pem"
KEY="/app/certs/key.pem"

if [ -f "$CERT" ] && [ -f "$KEY" ]; then
  echo "Gateway HTTPS sur le port 8443 (certificats trouvés)"
  exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8443 \
    --ssl-keyfile "$KEY" \
    --ssl-certfile "$CERT"
fi

echo "Gateway HTTP sur le port 8000 (pas de certificats — micro mobile limité)"
echo "Lancez: bash scripts/generate_ssl_certs.sh VOTRE_IP"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
