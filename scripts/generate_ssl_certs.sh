#!/usr/bin/env bash
# Certificat auto-signé pour HTTPS local (micro sur téléphone).
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
CERT_DIR="${DIR}/certs"
IP="${1:-}"

if [[ -z "$IP" ]]; then
  IP=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' | head -1)
fi
if [[ -z "$IP" ]]; then
  echo "Usage: $0 [IP_LAN]"
  echo "Exemple: $0 192.168.1.42"
  exit 1
fi

mkdir -p "$CERT_DIR"
CN="${IP}"
OPENSSL_CFG="${CERT_DIR}/openssl.cnf"

cat > "$OPENSSL_CFG" <<EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
x509_extensions = v3_req

[dn]
CN = ${CN}

[v3_req]
subjectAltName = @alt_names

[alt_names]
IP.1 = ${IP}
DNS.1 = localhost
IP.2 = 127.0.0.1
EOF

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout "${CERT_DIR}/key.pem" \
  -out "${CERT_DIR}/cert.pem" \
  -config "$OPENSSL_CFG"

echo ""
echo "Certificats créés dans ${CERT_DIR}/"
echo "Sur le téléphone, ouvrez : https://${IP}:8443/"
echo "Acceptez l’avertissement de sécurité (certificat auto-signé)."
echo ""
echo "Puis dans .env :"
echo "  LAN_PUBLIC_URL=https://${IP}:8443"
echo "  docker compose up -d gateway"
