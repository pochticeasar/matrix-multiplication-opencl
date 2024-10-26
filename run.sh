#!/bin/bash
set -x

TEST_FILE="0"
DEVICE_TYPE="dgpu"
WORK_LIMIT=1024

{
    # realization 0
    compile_command="clang *.c -O3 -lOpenCL -Wall -Wextra -o test.exe"
    run_command="./test.exe --input test_data/in${TEST_FILE}.txt --output out${TEST_FILE}.txt --realization 0"
    
    $compile_command 2>/dev/null

    if [ $? -eq 0 ]; then
        $run_command > .tmp
        if [ $? -eq 0 ]; then
            echo "Realization 0"
            cat .tmp
        else
            echo "Realization 0 fail"e
            exit
        fi
    fi

    echo    
}

{
    # realization 1
    compile_command="clang *.c -O3 -lOpenCL -Wall -Wextra -o test.exe"
    run_command="./test.exe --input test_data/in${TEST_FILE}.txt --output out${TEST_FILE}.txt --realization 1 --device-type ${DEVICE_TYPE} --device-index 0"
    
    $compile_command 2>/dev/null

    if [ $? -eq 0 ]; then
        $run_command > .tmp
        if [ $? -eq 0 ]; then
            echo "Realization 1"
            cat .tmp
        fi
    fi
    
    echo
}

{
    LOCAL2=(2 4 8 16 32)
    # realization 2
    run_command="./test.exe --input test_data/in${TEST_FILE}.txt --output out${TEST_FILE}.txt --realization 2 --device-type ${DEVICE_TYPE} --device-index 0"
    
    for local2 in "${LOCAL2[@]}"; do
        square=$((local2 * local2))

        if [ $square -gt $WORK_LIMIT ]; then
            continue
        fi

        compile_command="clang *.c -O3 -lOpenCL -Wall -Wextra -o test.exe -DLOCAL2=$local2"
        $compile_command 2>/dev/null

        if [ $? -eq 0 ]; then
            $run_command > .tmp
            if [ $? -eq 0 ]; then
                echo "Realization 2: $local2"
                cat .tmp
            fi
        fi

        echo
    done
}

{
    WI=(2 4 8 16)
    LOCAL3=(2 4 8 16 32 64 128)
    # realization 2
    run_command="./test.exe --input test_data/in${TEST_FILE}.txt --output out${TEST_FILE}.txt --realization 3 --device-type ${DEVICE_TYPE} --device-index 0"
    for wi in "${WI[@]}"; do
        for local3 in "${LOCAL3[@]}"; do
            square=$((local3 * local3 / wi))

            if [ $wi -gt $local3 ]; then
                continue
            fi

            mod=$((local3 % wi))

            if [ $mod -ne 0 ]; then
                continue
            fi

            if [ $square -gt $WORK_LIMIT ]; then
                continue
            fi
            compile_command="clang *.c -O3 -lOpenCL -Wall -Wextra -o test.exe -DLOCAL3=$local3 -DITEM=$wi"
            $compile_command 2>/dev/null

            if [ $? -eq 0 ]; then
                $run_command > .tmp
                if [ $? -eq 0 ]; then
                    echo "Realization 3: $local3 $wi"
                    cat .tmp
                fi
            fi

            echo
        done
    done
}
