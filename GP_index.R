



Run_PLINK = F
Run_PLOT = F
# Compile Sample information
source("../code/compileSampleInfo/compileSampleInfo.R")

# Convert to binary format
# system("plink --file GSA2016_308_025 --not-chr 0 --make-bed --out 1") # GWAS data from Rotterdam

# Visulize quality of the very basics
#source("../code/gwas_pipeline/GP_QC_Basic.R")


# Basic Marker QC
source("../code/gwas_pipeline/GP_QC_Basic_Marker_QC.R")

# Write an updated fam file with sex information
#write.table(data.frame(FID=dfSampleInfo$FID, IID=dfSampleInfo$RotterdamID, PID=0, MID=0, Sex=dfSampleInfo$Sex, PHE=dfSampleInfo$CaseControl), file="2.fam", quote = F, row.names = FALSE, col.names = FALSE, sep=" ")
write.table(data.frame(FID=dfSampleInfo$FID, IID=dfSampleInfo$RotterdamID, PID=0, MID=0, Sex=0, PHE=dfSampleInfo$CaseControl), file="2.fam", quote = F, row.names = FALSE, col.names = FALSE, sep=" ")

# QC of Samples
source("../code/gwas_pipeline/GP_QC_Samples.R")

# QC of Markers
source("../code/gwas_pipeline/GP_QC_Markers.R")

# Process the postQC dataset to fit GWAS analysis
source("../code/gwas_pipeline/GP_PostQC_Before_GWAS.R")

# GWAS
#source("../code/gwas_pipeline/GP_Results.R")

