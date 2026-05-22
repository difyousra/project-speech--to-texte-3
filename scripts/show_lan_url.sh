#!/usr/bin/env bash
# Affiche les URLs pour tester depuis un téléphone sur le même réseau Wi‑Fi.
set -euo pipefail

PORT="${GATEWAY_PORT:-8000}"

echo "=== Réclamation STT — accès réseau local ==="
echo ""

ips=()
if command -v hostname >/dev/null; then
  for ip in $(hostname -I 2>/dev/null); do
    [[ "$ip" =~ ^192\.168\. ]] || [[ "$ip" =~ ^10\. ]] || [[ "$ip" =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\. ]] && ips+=("$ip")
  done
fi

if [[ ${#ips[@]} -eq 0 ]]; then
  ip=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' | head -1)
  [[ -n "${ip:-}" ]] && ips+=("$ip")
fi

# Exclure interfaces Docker-only si une IP Wi‑Fi existe
wifi_ips=()
docker_ips=()
for ip in "${ips[@]}"; do
  if [[ "$ip" =~ ^172\.(1[7-9]|2[0-9]|3[0-1])\. ]]; then
    docker_ips+=("$ip")
  else
    wifi_ips+=("$ip")
  fi
done
if [[ ${#wifi_ips[@]} -gt 0 ]]; then
  ips=("${wifi_ips[@]}")
fi

if [[ ${#ips[@]} -eq 0 ]]; then
  echo "Impossible de détecter l’IP LAN. Essayez : ip -4 addr show wlan0"
  exit 1
fi

echo "Sur votre téléphone (même Wi‑Fi) — MICRO : utilisez HTTPS :"
for ip in "${ips[@]}"; do
  echo "  → https://${ip}:8443/   (recommandé)"
  echo "     http://${ip}:${PORT}/   (micro souvent bloqué)"
done
echo ""
echo "Générer le certificat SSL (une fois) :"
echo "  bash scripts/generate_ssl_certs.sh ${ips[0]}"
echo "  echo 'LAN_PUBLIC_URL=https://${ips[0]}:8443' >> .env"
echo "  docker compose up -d --build gateway"
echo ""
echo "Pare-feu : sudo ufw allow ${PORT}/tcp && sudo ufw allow 8443/tcp"
