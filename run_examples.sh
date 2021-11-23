# This file shows examples of how to synthesise, relight, and interpolate materials. 

declare -a arr=("leather_antique" "metal_black" "0281" "0282" "0284" "0285" "0286" "2000" "2003" "2004" "2007" "2018" "fabric_yellow" 
"fabric_zigzag" "leather_amber" "leather_antique" "leather_black" "metal_black" "metal_galvanized" "metal_rust" "paint_black" "paint_white" 
"plastic_red" "tape_silver" "tissue" "leather_amber" "wallpaper3" "wallpaper4" "wood_laminate" "2026" "2038" "2043" "fabric_weave" "book_leather_red" 
"cardboard" "fabric_orange" "metal_gritty")


for i in "${arr[@]}"
do
    # adjust paramterers to your needs
    python scripts/test.py --model trainings/Neuralmaterial --test_image_id "$i" --h 2048 --w 2048 --device cuda
done

# for i in "${arr[@]}"
# do
#     python scripts/relighting.py --model trainings/Neuralmaterial --test_image_id "$i"
# done


# python scripts/interpolate.py --model trainings/Neuralmaterial --weights1 metal_black --weights2 2004 --test_image_id1 metal_black --test_image_id2 2004
# python scripts/interpolate.py --model trainings/Neuralmaterial --weights1 fabric_yellow --weights2 2043 --test_image_id1 fabric_yellow --test_image_id2 2043
# python scripts/interpolate.py --model trainings/Neuralmaterial --weights1 0281 --weights2 fabric_zigzag --test_image_id1 0281 --test_image_id2 fabric_zigzag
# python scripts/interpolate.py --model trainings/Neuralmaterial --weights1 metal_gritty --weights2 2038 --test_image_id1 metal_gritty --test_image_id2 2038
# python scripts/interpolate.py --model trainings/Neuralmaterial --weights1 wood_laminate --weights2 0282 --test_image_id1 wood_laminate --test_image_id2 0282