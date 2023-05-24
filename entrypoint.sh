#!/bin/bash

GITHUB_REPO=https://github.com/stanford-futuredata/stk
ACCESS_TOKEN=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

cd /home/ci/actions-runner
./config.sh --url ${GITHUB_REPO} --pat ${ACCESS_TOKEN}

cleanup() {
    echo "Removing runner..."
    ./config.sh remove --unattended --pat ${ACCESS_TOKEN}
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

./run.sh & wait $!