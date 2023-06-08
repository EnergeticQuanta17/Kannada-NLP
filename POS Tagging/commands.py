import sys
import subprocess

args = int(sys.argv[1])
print(args)

if(args==1):
    subprocess.run(['python', 'create_indices.py', '--train', 'train.data', '--lang', 'kan'])
elif(args==2):
    subprocess.run(['python', 'train_BiLSTM.py', 
        '--train', 'train.data', 
        '--val', 'validation.data', 
        '--test', 'test.data', 
        '--embed', "C:/Users/student/Downloads/cc.kn.300.vec/cc.kn.300.vec", 
        '--lang', 'kan', 
        '--wt', 'weight-kan-bilstm-c2w', 
        '--epoch', '1']
    )
elif(args==3):
    subprocess.run(['python', 'predict_tags_using_model_and_generators.py', 
        '--test', 'test.data', 
        '--embed', "C:/Users/student/Downloads/cc.kn.300.vec/cc.kn.300.vec", 
        '--lang', 'kan', 
        '--model', 'weight-kan-bilstm-c2w', 
        '--output', 'output_file.txt']
    )