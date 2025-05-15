#!/bin/bash
echo "Generating figure 5 reidentification accuracy accross different device states..."
cd figure_5 && python3 main.py && echo "Done" && cd ..

echo "Generating figure 6 reidentification accuracy over time..."
cd figure_6 && python3 main.py && echo "Done" && cd ..

echo "Generating table 1 accuracy table..."
cd table_1 && python3 main.py && echo "Done" && cd ..