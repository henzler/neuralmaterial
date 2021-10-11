python scripts/test.py --model trainings/Neuralmaterial --test_image_id leather_antique 
python scripts/test.py --model trainings/Neuralmaterial --test_image_id metal_black
python scripts/test.py --model trainings/Neuralmaterial --test_image_id 0282
python scripts/test.py --model trainings/Neuralmaterial --test_image_id 0284
CUDA_VISIBLE_DEVICES=1 python scripts/test.py --model trainings/Neuralmaterial --test_image_id 2004 --h 2048 --w 2048

python scripts/interpolate.py --model trainings/Neuralmaterial --weights1 latest --weights2 latest --test_image_id1 0280 --test_image_id2 0281
python scripts/interpolate.py --model trainings/Neuralmaterial --weights1 0282 --weights2 0284 --test_image_id1 0282 --test_image_id2 0284