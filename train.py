import os
import torch
import copy
# from tqdm import tqdm
import time
from IPython.display import clear_output
from cod.tools import top_k_corrects, ConfusionMatrix, deleat_old_model

def train_model(model, 
                classification_criterion, 
                optimizer, 
                dataloaders,
                scheduler,
                batch_size, 
                snp_path, 
                num_epochs,
                top_k,
                phases=['T', 'V'],
                removing_intermediate_model=True
                ):
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    Softmax = torch.nn.Softmax()
    best_model_the_loss_classification = {phase: model.state_dict() for phase in phases}
    best_model_the_acc_classification = {phase: model.state_dict() for phase in phases}
    best_acc=dict.fromkeys(phases)
    best_Loss_classification=dict.fromkeys(phases)            
    best_epoch_acc = dict.fromkeys(phases)
    best_epoch_classification = dict.fromkeys(phases)
    epoch_acc=None
    pihati = ""
    model.to(devices)
    for epoch in range(num_epochs):
        pihati += 'Epoch {}/{}'.format(epoch + 1, num_epochs) + "\n"
        pihati += '-' * 10 + "\n"
        clear_output(wait=True)
        if epoch > 1:
            print('{} Loss: {:.4f} Acc: {:.4f}'.format("V", epoch_classification_loss, epoch_acc))
        print(pihati)
        for phase in phases:
            if phase == 'T':
                model.train()  # Установить модель в режим обучения
            else:
                model.eval()   #Установить модель в режим оценки
            # Обнуление параметров
            running_classification_loss = 0.0
            running_corrects = 0
            dataset_sizes=0
            confusion_matrix = ConfusionMatrix() 
            # pbar = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]), 
            #             desc="Epocha "+phase+" "+str(epoch+1)+"/"+str(num_epochs))
            iiter = 0
            for inputs, labels in dataloaders[phase]:
                # визуализация
                iiter += 1
                clear_output(wait=True)
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))
                print(iiter,"/",int(len(dataloaders[phase])))
                if iiter > 1:
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_classification_loss, epoch_acc))
                print()
                print(pihati)
                print(iiter,"/",int(len(dataloaders[phase])))
                # прогон через модель
                inputs = inputs.to(devices)
                labels = labels.to(devices)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'T'):
                    classification_out = model(inputs)
                    if dataloaders[phase].dataset.is_dog_cat:
                        classification_out = Softmax(classification_out)
                        n = dataloaders[phase].dataset.last_cat
                        classification_out = torch.cat((torch.sum(classification_out[:,n+1:],axis = 1).view(-1,1),torch.sum(classification_out[:,:n],axis = 1).view(-1,1)),axis = 1).to(devices)
                    total_classification_loss =  classification_criterion(classification_out, labels.to(dtype=torch.long))   
                    if phase == 'T':
                        # Вычислить градиенты
                        total_classification_loss.backward()
                        # Обновить веса
                        optimizer.step()
                confusion_matrix.apdate(classification_out,labels)        
                running_corrects+=top_k_corrects(classification_out,labels,top_k[phase])
                running_classification_loss += total_classification_loss.item() * inputs.size(0)          
                dataset_sizes += batch_size
                
                epoch_classification_loss = running_classification_loss / dataset_sizes
                epoch_acc = running_corrects / dataset_sizes

                # mem = torch.cuda.memory_reserved() / 1E9 if  torch.cuda.is_available() else 0
                # current_lr = optimizer.param_groups[0]['lr']
                # pbar.set_postfix(loss=f'{epoch_classification_loss:0.4f}',
                #                 acc=f'{epoch_acc:0.5f}',
                #                 lr=f'{current_lr:0.10f}',
                #                 gpu_memory=f'{mem:0.2f} GB')

            pihati += '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_classification_loss, epoch_acc)
            pihati += "\n"
            clear_output(wait=True)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_classification_loss, epoch_acc))
            print(pihati)

            if epoch==0:
                if phase == 'T':
                    with open(os.path.join(snp_path +"log.csv"), "w") as rez_file:
                        open_log_str='Epoch'
                        for log_phase in phases:
                            open_log_str=open_log_str+","+log_phase+"_loss,"+log_phase+'_acc'
                            if log_phase==phases[-1]:
                                open_log_str=open_log_str+'\n'
                        rez_file.write(open_log_str)
                else:
                    best_acc[phase]=epoch_acc
                    best_Loss_classification[phase]=epoch_classification_loss            
                    best_epoch_acc[phase] = 0
                    best_epoch_classification[phase] = 0

            # Обновить скорость обучения
            if phase == phases[0]:
                with open(os.path.join(snp_path+"log.csv"), "a") as rez_file:
                    rez_file.write(str(epoch+1)+','+str(round(epoch_classification_loss,4))+','+str(epoch_acc))
            elif phase == phases[-1]:
                with open(os.path.join(snp_path+"log.csv"), "a") as rez_file:
                    rez_file.write(','+str(round(epoch_classification_loss,4))+','+str(round(epoch_acc,4))+'\n')    
            else:
                with open(os.path.join(snp_path+"log.csv"), "a") as rez_file:
                    rez_file.write(','+str(round(epoch_classification_loss,4))+','+str(round(epoch_acc,4)))    
            
            if phase == 'T':
                if type(scheduler)!=torch.optim.lr_scheduler.ReduceLROnPlateau:
                    scheduler.step()
            elif phase == phases[-1]: 
                if type(scheduler)==torch.optim.lr_scheduler.ReduceLROnPlateau:
                    scheduler.step(epoch_acc)
            
            if phase != 'T' and best_acc[phase] < epoch_acc:
                best_acc[phase]=epoch_acc
                best_epoch_acc[phase]=epoch + 1
                best_model_the_acc_classification[phase] = model.state_dict()
                save_name = snp_path + str(best_epoch_acc[phase]) + "_"+phase+'_ACC top'+str(top_k[phase])+'-' + str(round(best_acc[phase],4)) + '_checkpoint.tar'
                if removing_intermediate_model ==True:
                    deleat_old_model('./'+snp_path,'ACC')
                torch.save({
                            'state_dict': model.state_dict(),
                            'confusion_matrix': confusion_matrix.get(),
                            }, save_name)
            if phase != 'T' and epoch_classification_loss < best_Loss_classification[phase]:
                best_Loss_classification[phase] = epoch_classification_loss
                best_epoch_classification[phase]=epoch + 1
                best_model_the_loss_classification[phase] = model.state_dict()
                save_name = snp_path + str(best_epoch_classification[phase]) + "_"+phase+'_CrossEntropyLoss-' + str(round(best_Loss_classification[phase],4)) + '_checkpoint.tar'
                if removing_intermediate_model ==True:
                    deleat_old_model('./'+snp_path,'Loss')
                torch.save({
                            'state_dict': model.state_dict(),
                            'confusion_matrix': confusion_matrix.get(),
                            }, save_name)
                
    # Конечное время и печать времени работы
    clear_output(wait=False)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    for phase in phases:
        if phase != 'T':
            print('Best {} Loss: {:.4f}, epoch {:.0f}  '.format(phase,best_Loss_classification[phase], best_epoch_classification[phase]))
            print('Best {} Acc : {:.4f}, epoch {:.0f}'.format(phase,best_acc[phase],best_epoch_acc[phase]))          
    print(pihati)
    overfit_model = model
    
    return best_model_the_loss_classification,best_model_the_acc_classification,overfit_model                             
                
                                
                
            

            
                