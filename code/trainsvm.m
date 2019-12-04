clc
clear

path = "/Users/zouyao/Documents/NEUfiles/MachineLearning/final project/train.csv";
testpath = "./final project/test.csv";
data = importdata(path);
traindata = data;
datatest = importdata(testpath);
sliddata1 = slidingData(traindata(:,1:14),4);
sliddata2 = slidingData(traindata(:,1:14),8);
sliddata3 = slidingData(traindata(:,1:14),16);

testsliddata1 = slidingData(datatest.data(:,2:15),4);
testsliddata2 = slidingData(datatest.data(:,2:15),8);
testsliddata3 = slidingData(datatest.data(:,2:15),16);

maxnode = [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000];
averageErrporsingle =zeros(1,size(maxnode,2));
averageErrporFSsingle = zeros(1,size(maxnode,2));


%%% optimize the hype-parameters
Mdl = fitcsvm(sliddata3(:,:),traindata(:,15),'OptimizeHyperparameters','auto','KernelFunction','rbf');


index =1;
%%%%%predict test with the optimized hyper-parameters
 for j = 1:size(sliddata1,3)
     
         svmmodel = fitcsvm(sliddata1(:,:,j),traindata(:,15),'KernelFunction','rbf','KernelScale',Mdl.hyperparamters.scale);
        testpredictlabel(:,index) = predict(svmmodel,testsliddata1(:,:,j));
        index= index+1;
 end

  for j = 1:size(sliddata2,3)
        svmmodel = fitcsvm(sliddata2(:,:,j),traindata(:,15),'KernelFunction','rbf','KernelScale',Mdl.hyperparamters.scale);
        testpredictlabel(:,index) = predict(svmmodel,testsliddata2(:,:,j));
        index = index+1;
  end
  svmmodel = fitcsvm(sliddata3(:,:),traindata(:,15),'KernelFunction','rbf','KernelScale',Mdl.hyperparamters.scale);
  testpredictlabel(:,index) = predict(svmmodel,testsliddata3(:,:));
  
  for k = 1:4980
        testpredictlabel(k,20) = sum(testpredictlabel(k,:));
        if testpredictlabel(k,20)>=10
            testpredictlabel(k,20) = 1;
        else
            testpredictlabel(k,20) = 0;
        end
  end