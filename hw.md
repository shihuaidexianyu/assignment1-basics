# hw

## Problem (train_bpe_tiny stories): BPE Training on Tiny Stories (2 points)

### a

I tarin my bpe on TinyStoriesV2-GPT4-train. The resource is following:

```text
❯ /usr/bin/time -v uv run /home/hw/learn/assignment1-basics/train.py --input /home/hw/learn/assignment1-basics/data/data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000 --num-workers 16 --out-dir /home/hw/learn/assignment1-basics/tokenizer_out_10000
--- Starting BPE Training on /home/hw/learn/assignment1-basics/data/data/TinyStoriesV2-GPT4-train.txt ---
Divided file into 16 chunks. Running pre-tokenization...
Pre-tokenization complete. Unique words: 59933
Starting Merge Iterations...
Merge 1/9743: (32, 116) -> 256 (freq: 63482199)
Merge 101/9743: (319, 352) -> 356 (freq: 3023009)
Merge 201/9743: (265, 452) -> 456 (freq: 1406956)
Merge 301/9743: (258, 103) -> 556 (freq: 789458)
Merge 401/9743: (273, 263) -> 656 (freq: 521684)
Merge 501/9743: (295, 597) -> 756 (freq: 369517)
Merge 601/9743: (272, 305) -> 856 (freq: 278371)
Merge 701/9743: (288, 317) -> 956 (freq: 220974)
Merge 801/9743: (115, 111) -> 1056 (freq: 177517)
Merge 901/9743: (518, 301) -> 1156 (freq: 147517)
Merge 1001/9743: (277, 320) -> 1256 (freq: 127159)
Merge 1101/9743: (665, 352) -> 1356 (freq: 107974)
Merge 1201/9743: (276, 856) -> 1456 (freq: 92187)
Merge 1301/9743: (116, 290) -> 1556 (freq: 80291)
Merge 1401/9743: (276, 439) -> 1656 (freq: 68380)
Merge 1501/9743: (626, 642) -> 1756 (freq: 59779)
Merge 1601/9743: (418, 1617) -> 1856 (freq: 53737)
Merge 1701/9743: (278, 593) -> 1956 (freq: 47983)
Merge 1801/9743: (304, 116) -> 2056 (freq: 43399)
Merge 1901/9743: (288, 306) -> 2156 (freq: 38893)
Merge 2001/9743: (687, 418) -> 2256 (freq: 35430)
Merge 2101/9743: (67, 422) -> 2356 (freq: 32599)
Merge 2201/9743: (277, 1662) -> 2456 (freq: 30199)
Merge 2301/9743: (543, 1299) -> 2556 (freq: 27604)
Merge 2401/9743: (373, 1101) -> 2656 (freq: 25263)
Merge 2501/9743: (2624, 593) -> 2756 (freq: 23441)
Merge 2601/9743: (354, 110) -> 2856 (freq: 21869)
Merge 2701/9743: (633, 2904) -> 2956 (freq: 20508)
Merge 2801/9743: (276, 2149) -> 3056 (freq: 19442)
Merge 2901/9743: (820, 298) -> 3156 (freq: 18367)
Merge 3001/9743: (962, 287) -> 3256 (freq: 17389)
Merge 3101/9743: (478, 1557) -> 3356 (freq: 16609)
Merge 3201/9743: (331, 2534) -> 3456 (freq: 15869)
Merge 3301/9743: (3541, 832) -> 3556 (freq: 15110)
Merge 3401/9743: (738, 743) -> 3656 (freq: 14415)
Merge 3501/9743: (283, 3747) -> 3756 (freq: 13574)
Merge 3601/9743: (277, 1596) -> 3856 (freq: 13005)
Merge 3701/9743: (2220, 605) -> 3956 (freq: 12445)
Merge 3801/9743: (3259, 367) -> 4056 (freq: 11926)
Merge 3901/9743: (4118, 101) -> 4156 (freq: 11336)
Merge 4001/9743: (798, 791) -> 4256 (freq: 10785)
Merge 4101/9743: (3438, 1009) -> 4356 (freq: 10129)
Merge 4201/9743: (1096, 304) -> 4456 (freq: 9460)
Merge 4301/9743: (1848, 675) -> 4556 (freq: 8922)
Merge 4401/9743: (1220, 281) -> 4656 (freq: 8304)
Merge 4501/9743: (1899, 115) -> 4756 (freq: 7624)
Merge 4601/9743: (2269, 1316) -> 4856 (freq: 7111)
Merge 4701/9743: (4257, 2198) -> 4956 (freq: 6600)
Merge 4801/9743: (3619, 1308) -> 5056 (freq: 6164)
Merge 4901/9743: (259, 105) -> 5156 (freq: 5736)
Merge 5001/9743: (1990, 115) -> 5256 (freq: 5349)
Merge 5101/9743: (1332, 463) -> 5356 (freq: 4935)
Merge 5201/9743: (1297, 98) -> 5456 (freq: 4545)
Merge 5301/9743: (2829, 2177) -> 5556 (freq: 4225)
Merge 5401/9743: (268, 320) -> 5656 (freq: 3885)
Merge 5501/9743: (264, 5731) -> 5756 (freq: 3589)
Merge 5601/9743: (2131, 3841) -> 5856 (freq: 3307)
Merge 5701/9743: (4486, 1671) -> 5956 (freq: 3118)
Merge 5801/9743: (894, 1115) -> 6056 (freq: 2929)
Merge 5901/9743: (483, 5182) -> 6156 (freq: 2711)
Merge 6001/9743: (5960, 281) -> 6256 (freq: 2539)
Merge 6101/9743: (2739, 3799) -> 6356 (freq: 2394)
Merge 6201/9743: (3047, 115) -> 6456 (freq: 2237)
Merge 6301/9743: (829, 6546) -> 6556 (freq: 2100)
Merge 6401/9743: (1190, 6647) -> 6656 (freq: 1995)
Merge 6501/9743: (1432, 298) -> 6756 (freq: 1880)
Merge 6601/9743: (276, 300) -> 6856 (freq: 1789)
Merge 6701/9743: (104, 753) -> 6956 (freq: 1680)
Merge 6801/9743: (4543, 517) -> 7056 (freq: 1598)
Merge 6901/9743: (4514, 263) -> 7156 (freq: 1520)
Merge 7001/9743: (360, 6738) -> 7256 (freq: 1437)
Merge 7101/9743: (596, 646) -> 7356 (freq: 1365)
Merge 7201/9743: (265, 601) -> 7456 (freq: 1288)
Merge 7301/9743: (1426, 114) -> 7556 (freq: 1221)
Merge 7401/9743: (3318, 115) -> 7656 (freq: 1157)
Merge 7501/9743: (1924, 1900) -> 7756 (freq: 1095)
Merge 7601/9743: (277, 7184) -> 7856 (freq: 1048)
Merge 7701/9743: (7258, 263) -> 7956 (freq: 1003)
Merge 7801/9743: (72, 6064) -> 8056 (freq: 967)
Merge 7901/9743: (283, 98) -> 8156 (freq: 918)
Merge 8001/9743: (4698, 4430) -> 8256 (freq: 878)
Merge 8101/9743: (264, 1525) -> 8356 (freq: 840)
Merge 8201/9743: (462, 105) -> 8456 (freq: 809)
Merge 8301/9743: (836, 115) -> 8556 (freq: 778)
Merge 8401/9743: (971, 1492) -> 8656 (freq: 747)
Merge 8501/9743: (296, 670) -> 8756 (freq: 715)
Merge 8601/9743: (2468, 118) -> 8856 (freq: 690)
Merge 8701/9743: (2381, 1269) -> 8956 (freq: 660)
Merge 8801/9743: (416, 298) -> 9056 (freq: 632)
Merge 8901/9743: (259, 1984) -> 9156 (freq: 608)
Merge 9001/9743: (510, 9254) -> 9256 (freq: 584)
Merge 9101/9743: (703, 1593) -> 9356 (freq: 561)
Merge 9201/9743: (256, 8405) -> 9456 (freq: 542)
Merge 9301/9743: (1599, 301) -> 9556 (freq: 521)
Merge 9401/9743: (1049, 1784) -> 9656 (freq: 502)
Merge 9501/9743: (76, 274) -> 9756 (freq: 485)
Merge 9601/9743: (1111, 516) -> 9856 (freq: 469)
Merge 9701/9743: (2279, 445) -> 9956 (freq: 454)
Longest token length after training: 15 bytes
Training Complete.
Saved vocab to /home/hw/learn/assignment1-basics/tokenizer_out_10000/tokenizer_vocab.json
Saved merges to /home/hw/learn/assignment1-basics/tokenizer_out_10000/tokenizer_merges.txt
        Command being timed: "uv run /home/hw/learn/assignment1-basics/train.py --input /home/hw/learn/assignment1-basics/data/data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000 --num-workers 16 --out-dir /home/hw/learn/assignment1-basics/tokenizer_out_10000"
        User time (seconds): 2158.09
        System time (seconds): 13.61
        Percent of CPU this job got: 279%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 12:56.48
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 713444
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 0
        Minor (reclaiming a frame) page faults: 2603909
        Voluntary context switches: 1062
        Involuntary context switches: 211627
        Swaps: 0
        File system inputs: 0
        File system outputs: 536
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0
```

### b
