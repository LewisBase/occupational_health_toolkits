# -*- coding: utf-8 -*-
"""
@DATE: 2024-01-08 20:34:10
@Author: Liu Hengjiang
@File: training/multi_task_trainer.py
@Software: vscode
@Description:
        多任务学习模型训练器
"""
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def train_model(model, train_loader, val_loader, epoch, loss_function, optimizer, path, early_stop):
    """
    多任务学习模型训练函数
    
    Args:
        model: PyTorch 多任务模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epoch: 训练轮数
        loss_function: 损失函数
        optimizer: 优化器
        path: 模型保存路径模板
        early_stop: 早停轮数
        
    Returns:
        None
    """
    # use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 多少步内验证集的loss没有变小就提前停止
    patience, eval_loss = 0, 0
    
    # train
    for i in range(epoch):
        y_train_income_true = []
        y_train_income_predict = []
        y_train_marry_true = []
        y_train_marry_predict = []
        total_loss, count = 0, 0
        for idx, (x, y1, y2) in tqdm(enumerate(train_loader), total=len(train_loader)):
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            predict = model(x)
            y_train_income_true += list(y1.squeeze().cpu().numpy())
            y_train_marry_true += list(y2.squeeze().cpu().numpy())
            y_train_income_predict += list(predict[0].squeeze().cpu().detach().numpy())
            y_train_marry_predict += list(predict[1].squeeze().cpu().detach().numpy())
            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss = loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            count += 1
        torch.save(model, path.format(i + 1))
        income_auc = roc_auc_score(y_train_income_true, y_train_income_predict)
        marry_auc = roc_auc_score(y_train_marry_true, y_train_marry_predict)
        logger.info(f"Epoch {i + 1} train loss is {total_loss / count:.3f}, income auc is {income_auc:.3f} and marry auc is {marry_auc:.3f}")
        
        # 验证
        total_eval_loss = 0
        model.eval()
        count_eval = 0
        y_val_income_true = []
        y_val_marry_true = []
        y_val_income_predict = []
        y_val_marry_predict = []
        for idx, (x, y1, y2) in tqdm(enumerate(val_loader), total=len(val_loader)):
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            predict = model(x)
            y_val_income_true += list(y1.squeeze().cpu().numpy())
            y_val_marry_true += list(y2.squeeze().cpu().numpy())
            y_val_income_predict += list(predict[0].squeeze().cpu().detach().numpy())
            y_val_marry_predict += list(predict[1].squeeze().cpu().detach().numpy())
            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss = loss_1 + loss_2
            total_eval_loss += float(loss)
            count_eval += 1
        income_auc = roc_auc_score(y_val_income_true, y_val_income_predict)
        marry_auc = roc_auc_score(y_val_marry_true, y_val_marry_predict)
        logger.info(f"Epoch {i + 1} val loss is {total_eval_loss / count_eval:.3f}, income auc is {income_auc:.3f} and marry auc is {marry_auc:.3f}")
        
        # earl stopping
        if i == 0:
            eval_loss = total_eval_loss / count_eval
        else:
            if total_eval_loss / count_eval < eval_loss:
                eval_loss = total_eval_loss / count_eval
            else:
                if patience < early_stop:
                    patience += 1
                else:
                    logger.info(f"val loss is not decrease in {patience} epoch and break training")
                    break
