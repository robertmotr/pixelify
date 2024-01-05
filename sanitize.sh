#!/bin/bash
compute-sanitizer --check-device-heap yes --check-exit-code yes --print-session-details --print-level info build/tests
compute-sanitizer --check-device-heap yes --check-exit-code yes --print-session-details --print-level info --tool=racecheck build/tests
compute-sanitizer --check-device-heap yes --check-exit-code yes --print-session-details --print-level info --tool=initcheck build/tests
compute-sanitizer --check-device-heap yes --check-exit-code yes --print-session-details --print-level info --tool=synccheck build/tests
