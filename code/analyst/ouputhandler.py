import os

from analyst.viewhandler import view_common_predict


def out4LatexTableItem(workspace, station, preName, template):
    cnnworkspace = workspace + 'proc\\cnn'
    rfrworkspace = workspace + 'proc\\rfr'
    svrrbfworkspace = workspace + 'proc\\svrrbf'
    template = template.replace('station', station)
    for i in range(1, 4):
        predictname = preName + str(i)
        cnn = view_common_predict(cnnworkspace, predictname)
        rfr = view_common_predict(rfrworkspace, predictname)
        svr = view_common_predict(svrrbfworkspace, predictname)
        if i == 1:
            template = template.replace('raw-n', '%.2f' % (cnn[0]))
            template = template.replace('cnn-n', '%.2f' % (cnn[1]))
            template = template.replace('rfr-n', '%.2f' % (rfr[1]))
            template = template.replace('svr-n', '%.2f' % (svr[1]))
            if cnn[2] > 0:
                template = template.replace('cnn-p-n', '%.0f\\textcolor{green}{$\\uparrow$}' % (cnn[2]))
            else:
                template = template.replace('cnn-p-n', '%.0f\\textcolor{red}{$\\downarrow$}' % (cnn[2]))
            if rfr[2] > 0:
                template = template.replace('rfr-p-n', '%.0f\\textcolor{green}{$\\uparrow$}' % (rfr[2]))
            else:
                template = template.replace('rfr-p-n', '%.0f\\textcolor{red}{$\\downarrow$}' % (rfr[2]))
            if svr[2] > 0:
                template = template.replace('svr-p-n', '%.0f\\textcolor{green}{$\\uparrow$}' % (svr[2]))
            else:
                template = template.replace('svr-p-n', '%.0f\\textcolor{red}{$\\downarrow$}' % (svr[2]))
        elif i == 2:
            template = template.replace('raw-e', '%.2f' % (cnn[0]))
            template = template.replace('cnn-e', '%.2f' % (cnn[1]))
            template = template.replace('rfr-e', '%.2f' % (rfr[1]))
            template = template.replace('svr-e', '%.2f' % (svr[1]))
            if cnn[2] > 0:
                template = template.replace('cnn-p-e', '%.0f\\textcolor{green}{$\\uparrow$}' % (cnn[2]))
            else:
                template = template.replace('cnn-p-e', '%.0f\\textcolor{red}{$\\downarrow$}' % (cnn[2]))
            if rfr[2] > 0:
                template = template.replace('rfr-p-e', '%.0f\\textcolor{green}{$\\uparrow$}' % (rfr[2]))
            else:
                template = template.replace('rfr-p-e', '%.0f\\textcolor{red}{$\\downarrow$}' % (rfr[2]))
            if svr[2] > 0:
                template = template.replace('svr-p-e', '%.0f\\textcolor{green}{$\\uparrow$}' % (svr[2]))
            else:
                template = template.replace('svr-p-e', '%.0f\\textcolor{red}{$\\downarrow$}' % (svr[2]))
        elif i == 3:
            template = template.replace('raw-d', '%.2f' % (cnn[0]))
            template = template.replace('cnn-d', '%.2f' % (cnn[1]))
            template = template.replace('rfr-d', '%.2f' % (rfr[1]))
            template = template.replace('svr-d', '%.2f' % (svr[1]))
            if cnn[2] > 0:
                template = template.replace('cnn-p-d', '%.0f\\textcolor{green}{$\\uparrow$}' % (cnn[2]))
            else:
                template = template.replace('cnn-p-d', '%.0f\\textcolor{red}{$\\downarrow$}' % (cnn[2]))
            if rfr[2] > 0:
                template = template.replace('rfr-p-d', '%.0f\\textcolor{green}{$\\uparrow$}' % (rfr[2]))
            else:
                template = template.replace('rfr-p-d', '%.0f\\textcolor{red}{$\\downarrow$}' % (rfr[2]))
            if svr[2] > 0:
                template = template.replace('svr-p-d', '%.0f\\textcolor{green}{$\\uparrow$}' % (svr[2]))
            else:
                template = template.replace('svr-p-d', '%.0f\\textcolor{red}{$\\downarrow$}' % (svr[2]))
    return template
def outCnn4LatexTableItem(workspace, station, preName, template):
    cnnworkspace = workspace + 'proc\\cnn'
    template = template.replace('station', station)
    for i in range(1, 4):
        predictname = preName + str(i)
        cnn = view_common_predict(cnnworkspace, predictname)
        if i == 1:
            template = template.replace('raw-n', '%.2f' % (cnn[0]))
            template = template.replace('cnn-n', '%.2f' % (cnn[1]))
            if cnn[2] > 0:
                template = template.replace('cnn-p-n', '%.0f\\textcolor{green}{$\\uparrow$}' % (cnn[2]))
            else:
                template = template.replace('cnn-p-n', '%.0f\\textcolor{red}{$\\downarrow$}' % (cnn[2]))
        elif i == 2:
            template = template.replace('raw-e', '%.2f' % (cnn[0]))
            template = template.replace('cnn-e', '%.2f' % (cnn[1]))
            if cnn[2] > 0:
                template = template.replace('cnn-p-e', '%.0f\\textcolor{green}{$\\uparrow$}' % (cnn[2]))
            else:
                template = template.replace('cnn-p-e', '%.0f\\textcolor{red}{$\\downarrow$}' % (cnn[2]))
        elif i == 3:
            template = template.replace('raw-d', '%.2f' % (cnn[0]))
            template = template.replace('cnn-d', '%.2f' % (cnn[1]))
            if cnn[2] > 0:
                template = template.replace('cnn-p-d', '%.0f\\textcolor{green}{$\\uparrow$}' % (cnn[2]))
            else:
                template = template.replace('cnn-p-d', '%.0f\\textcolor{red}{$\\downarrow$}' % (cnn[2]))
    return template
def clearWorkspace(workspace):
    cnnworkspace = workspace + 'proc\\cnn'
    rfrworkspace = workspace + 'proc\\rfr'
    svrrbfworkspace = workspace + 'proc\\svrrbf'
    list_dirs = os.walk(cnnworkspace)
    delpath=[]
    for root, dirs, files in list_dirs:
        for d in dirs:
            delpath.append(os.path.join(root, d))
    list_dirs = os.walk(rfrworkspace)
    for root, dirs, files in list_dirs:
        for d in dirs:
            delpath.append(os.path.join(root, d))
    list_dirs = os.walk(svrrbfworkspace)
    for root, dirs, files in list_dirs:
        for d in dirs:
            delpath.append(os.path.join(root, d))
    delfiles=[]
    for dp in delpath:
        list_dirs = os.walk(dp)
        for root, dirs, files in list_dirs:
            for file in files:
                delfiles.append(os.path.join(root, file))
    for df in delfiles:
        os.remove(df)