# AfterQuery / Project Turing repository image.
# Adapted from the validated Node reference: pinned base (no :latest),
# pinned apt versions, apt-get install --no-install-recommends.
# Installs the toolchain `make test` uses: C11 compiler, Make, Python 3, NumPy.

FROM debian:bookworm-20251229-slim@sha256:d5d3f9c23164ea16f31852f95bd5959aad1c5e854332fe00f7b3a20fcc9f635c

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# Freeze apt to the same date as the base tag so the pinned versions stay
# resolvable after Debian security supersedes them.
RUN printf '%s\n' \
      'deb http://snapshot.debian.org/archive/debian/20251229T000000Z bookworm main' \
      'deb http://snapshot.debian.org/archive/debian-security/20251229T000000Z bookworm-security main' \
      > /etc/apt/sources.list \
    && printf 'Acquire::Check-Valid-Until "false";\n' \
      > /etc/apt/apt.conf.d/99snapshot \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
      ca-certificates=20230311+deb12u1 \
      gcc=4:12.2.0-3 \
      git=1:2.39.5-0+deb12u2 \
      libc6-dev=2.36-9+deb12u13 \
      make=4.3-4.1 \
      python3=3.11.2-1+b1 \
      python3-numpy=1:1.24.2-1+deb12u1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
