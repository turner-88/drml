#!/bin/sh
# Database helpers used by the Makefile. Run via scripts/with-env.sh so that
# .env is loaded first.
#
# Usage:
#   scripts/db.sh shell            open a MySQL shell on the app database
#   scripts/db.sh schema [FORCE]   create/reset the schema (destructive)

set -eu

DB_HOST=${DB_HOST:-127.0.0.1}
DB_PORT=${DB_PORT:-3306}
DB_USER=${DB_USER:-root}
DB_NAME=${DB_NAME:-drml}

# Connect over TCP rather than the unix socket: MariaDB authenticates local
# socket connections for root via unix_socket, which ignores DB_PASSWORD and
# rejects the login. The app connects over TCP too, so this matches it.
mysql_cmd() {
    if [ -n "${DB_PASSWORD:-}" ]; then
        mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" "-p$DB_PASSWORD" "$@"
    else
        mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" "$@"
    fi
}

require_connection() {
    if ! mysql_cmd -e "SELECT 1" >/dev/null 2>&1; then
        echo "error: cannot reach MySQL at $DB_HOST:$DB_PORT as '$DB_USER'." >&2
        echo "       check DB_HOST/DB_PORT/DB_USER/DB_PASSWORD in .env" >&2
        exit 1
    fi
}

case "${1:-}" in
shell)
    require_connection
    exec mysql_cmd "$DB_NAME"
    ;;

schema)
    require_connection

    # Refuse to wipe a database that holds real scans unless forced. This drops
    # every table, and these are patient records.
    rows=$(mysql_cmd -N -B "$DB_NAME" -e "SELECT COUNT(*) FROM scan" 2>/dev/null || echo 0)
    if [ "${rows:-0}" -gt 0 ] && [ "${FORCE:-}" != "1" ]; then
        echo "refusing to reset: database '$DB_NAME' holds $rows scan(s)." >&2
        echo "this drops every table. re-run with FORCE=1 if that is what you want." >&2
        exit 1
    fi

    mysql_cmd -e "CREATE DATABASE IF NOT EXISTS \`$DB_NAME\` \
        CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;"
    mysql_cmd "$DB_NAME" < internal/database/migration/schema.sql
    echo "schema applied to '$DB_NAME' at $DB_HOST:$DB_PORT"
    ;;

*)
    echo "usage: $0 {shell|schema}" >&2
    exit 2
    ;;
esac
