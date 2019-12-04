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

maxlevel = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300];
averageErrporsingle =zeros(1,size(maxlevel,2));
averageErrporFSsingle = zeros(1,size(maxlevel,2));


%%%%% using 10-fold cross-validation on the training data set to evaluate
%%%%% the model
for numlevel = 1:size(maxlevel,2)   %%%% iteration of the level
for i = 1:10
index = 1;
predictlabel = zeros(1000,20);
slice = ((i-1)*1000+1:i*1000);
    trainslice=[1:slice(1)-1,slice(end)+1:10000];
 for j = 1:size(sliddata1,3)
         t = templateTree('MaxNumSplits',500);
         ensammodel = fitcensemble(sliddata1(trainslice,:,j),traindata(trainslice,15),'Learners',t,'Method','AdaBoostM1','NumLearningCycles',maxlevel(numlevel));
         predictlabel(:,index) = predict(ensammodel,sliddata1(slice,:,j));
        index= index+1;
 end

  for j = 1:size(sliddata2,3)
         t = templateTree('MaxNumSplits',500);
         ensammodel = fitcensemble(sliddata2(trainslice,:,j),traindata(trainslice,15),'Learners',t,'Method','AdaBoostM1','NumLearningCycles',maxlevel(numlevel));
         predictlabel(:,index) = predict(ensammodel,sliddata2(slice,:,j));
        index = index+1;
  end
  
  t = templateTree('MaxNumSplits',500);
  ensammodel = fitcensemble(sliddata3(trainslice,:),traindata(trainslice,15),'Learners',t,'Method','AdaBoostM1','NumLearningCycles',maxlevel(numlevel));
  predictlabel(:,index) = predict(ensammodel,sliddata3(slice,:));
  index = index+1;

  
  for k = 1:1000
        predictlabel(k,index) = sum(predictlabel(k,:));
        if predictlabel(k,index)>=10
            predictlabel(k,index) = 1;
        else
            predictlabel(k,index) = 0;
        end
  end
   B1(:,:,i) = confusionmat(traindata(slice,15),predictlabel(:,index-1));
  B(:,:,i) = confusionmat(traindata(slice,15),predictlabel(:,index));
end





sumB1 =0;
sumB = 0;
for i = 1:10
  sumB1 = sumB1+( B1(1,2,i)+B1(2,1,i) )/1000;
  sumB = sumB+( B(1,2,i)+B(2,1,i) )/1000;
end

averageErrporsingle(numlevel) = sumB1/10;
averageErrporFSsingle(numlevel) = sumB/10;

end


plot(maxlevel,averageErrporsingle,'r');
hold on
plot(maxlevel,averageErrporFSsingle,'b');

xlabel('level of decision tree in Adaboost','FontSize',16);
ylabel('average error of cross-validation','FontSize',16)
legend('Adaboost','FS with Adaboost','FontSize',16)
title('adaboost with maxtreenode 500','FontSize',16)

index =1;
%%%%%predict test
 for j = 1:size(sliddata1,3)
         t = templateTree('MaxNumSplits',500);
         ensammodel = fitcensemble(sliddata1(:,:,j),traindata(:,15),'Learners',t,'Method','AdaBoostM1','NumLearningCycles',500);
        testpredictlabel(:,index) = predict(ensammodel,testsliddata1(:,:,j));
        index= index+1;
 end

  for j = 1:size(sliddata2,3)
         t = templateTree('MaxNumSplits',500);
         ensammodel = fitcensemble(sliddata1(:,:,j),traindata(:,15),'Learners',t,'Method','AdaBoostM1','NumLearningCycles',500);
        testpredictlabel(:,index) = predict(ensammodel,testsliddata2(:,:,j));
        index = index+1;
  end
  
  treemodel = fitctree(sliddata3(:,:),traindata(:,15));
  t = templateTree('MaxNumSplits',500);
  ensammodel = fitcensemble(sliddata1(:,:,j),traindata(:,15),'Learners',t,'Method','AdaBoostM1','NumLearningCycles',500);
  testpredictlabel(:,index) = predict(ensammodel,testsliddata3(:,:));
  

  
  for k = 1:4980
        testpredictlabel(k,20) = sum(testpredictlabel(k,:));
        if testpredictlabel(k,20)>=10
            testpredictlabel(k,20) = 1;
        else
            testpredictlabel(k,20) = 0;
        end
  end



