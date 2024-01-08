#!/bin/bash
compute-sanitizer --check-device-heap yes --check-exit-code yes --tool=memcheck build/tests
sleep 3
compute-sanitizer --check-device-heap yes --check-exit-code yes --tool=racecheck build/tests
sleep 3
compute-sanitizer --check-device-heap yes --check-exit-code yes --tool=initcheck build/tests
sleep 3
compute-sanitizer --check-device-heap yes --check-exit-code yes --tool=synccheck build/tests
sleep 3
valgrind --tool=memcheck --track-origins=yes --log-file=valgrind-memcheck.log build/tests
sleep 3
valgrind --tool=helgrind --log-file=valgrind-helgrind.log build/tests
