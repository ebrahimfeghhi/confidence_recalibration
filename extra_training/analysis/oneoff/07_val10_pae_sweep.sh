cd /home/ebrahim/confidence_recalibration/extra_training
for E in 3 5 10 21 60; do
  B=training/runs/335psh/resume_ep$E.pt
  [ -f "$B" ] || B=training/runs/2u8tf4/resume_ep$E.pt
  [ -f "$B" ] || { echo "no bundle for ep$E"; continue; }
  [ -d training/val10/lossonly_ep$E ] && { echo "ep$E already done"; continue; }
  bash analysis/oneoff/03_val10_loss_only.sh "$B" "ep$E" 7 > /dev/null 2>&1
  echo "ep$E done"
done
