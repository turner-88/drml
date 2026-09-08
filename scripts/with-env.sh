#!/bin/sh
# Load .env, then exec the given command.
#
# Make does not read .env, but the app does (via godotenv), so Makefile targets
# that talk to the database would otherwise use a different configuration than
# the server. This keeps the two in sync.
#
# Precedence matches godotenv: variables already present in the environment win,
# so `DB_NAME=other scripts/with-env.sh ...` overrides the file rather than
# being silently replaced by it.
#
# Usage: scripts/with-env.sh <command> [args...]

set -eu

ENV_FILE="${ENV_FILE:-.env}"

if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip blanks and comments.
        case "$line" in
            ''|'#'*) continue ;;
        esac

        key=${line%%=*}
        value=${line#*=}

        # A line with no '=' is not an assignment.
        [ "$key" = "$line" ] && continue

        # Trim surrounding whitespace and an optional `export ` prefix.
        key=$(printf '%s' "$key" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//; s/^export[[:space:]]\{1,\}//')
        case "$key" in
            ''|*[!A-Za-z0-9_]*) continue ;;   # not a valid shell identifier
        esac

        # Strip one layer of matching quotes.
        case "$value" in
            \"*\") value=$(printf '%s' "$value" | sed 's/^"//; s/"$//') ;;
            \'*\') value=$(printf '%s' "$value" | sed "s/^'//; s/'$//") ;;
        esac

        # Only set what the caller has not already provided.
        eval "current=\${$key-}"
        [ -n "${current:-}" ] || export "$key=$value"
    done < "$ENV_FILE"
fi

exec "$@"
