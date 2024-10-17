#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <TCanvas.h>
#include <TH1F.h>
#include <TFile.h>
#include <TString.h>
#include <TSystem.h>
#include <TROOT.h>


vector<double> load_txt_file(const string& directory, int col, int rows_to_draw); 
double langaufun(double *x, double *par);

const int NBIN=50;
const int MAXTOT=25;


void Draw_ToT_certain(std::string directory = 
    //"/Users/yoonha/cernbox/A-STEP/241009/ToT_distributions_THR200_APS3-W08-S03_20241010_094016") {
    "/Users/yoonha/cernbox/A-STEP/241011/ToT_distributions_THR200_APS3-W08-S11_20241011_170850") {

    std::vector<int> cols_to_draw;  // Default columns
    std::vector<int> rows_to_draw;  // Default rows_to_draw

    cols_to_draw = {15, 16, 17, 18, 19};
    rows_to_draw = {24, 23, 22, 21, 20};

    //cols_to_draw = {5, 6, 7, 8, 9};
    //rows_to_draw = {4, 3, 2, 1, 0};

    const int nc = cols_to_draw.size();  // Number of columns
    const int nr = rows_to_draw.size();  // Number of rows_to_draw


    std::string file_name = gSystem->BaseName(directory.c_str());
    /*
    std::ofstream file( ( "./" + file_name + ".csv").c_str() );
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
        return;
    }
    */
    // Create a histogram to hold all the values
    TH1F* hist_tot_all = new TH1F("hist_tot_all", "Total Histogram", NBIN, 0, MAXTOT);
    TH1F* hist_tot_pixel[35][35];
    TF1* fit_tot_pixel[35][35];

    TH1F* hist_MPV = new TH1F("hist_MPV", "MPV Distribution", 50, 0, 10);

    TH2F* h2_hit = new TH2F("h2_hit", "h2_hit", 35, 0, 35, 35, 0, 35);
    TH2F* h2_MPV = new TH2F("h2_MPV", "h2_MPV", 35, 0, 35, 35, 0, 35);
    TH2F* h2_chisqrndf = new TH2F("h2_chisqrndf", "h2_chisqrndf", 35, 0, 35, 35, 0, 35);


    //file << "col,row,nhits,MPV,chi2/ndf" << std::endl;
    float MPV[35][35];

    // Width, MP, Area, GSigma
    double pllo[4] = {0.09, 0.5, 5, 0.2};
    double plhi[4] = {2.80, 20.0, 90000, 1.8};
    double sv[4] = {0.5, 4.0, 500, 0.4};

    double fp[4], fpe[4];

    for (int r : rows_to_draw) {
        for (int c : cols_to_draw) {
            // Load data for the current column and rows_to_draw
            std::vector<double> data = load_txt_file(directory, c, r);

            hist_tot_pixel[c][r] = new TH1F(Form("c%d_r%d", c, r), Form("c%d_r%d", c, r), NBIN, 0, MAXTOT);

            if (data.empty()){
                continue;
            }

            int hit = 0;
            for (double value : data)
            {
                hist_tot_pixel[c][r]->Fill(value);
                hist_tot_all->Fill(value); // Fill the total histogram
                hit++;
            }

            fit_tot_pixel[c][r] = new TF1(Form("langau_c%d_r%d", c, r), langaufun, 2, 10, 4);
            fit_tot_pixel[c][r] -> SetParNames("Width","MP","Area","GSigma");
            fit_tot_pixel[c][r] -> SetParameters(sv);
            for(int par=0; par<4; par++) fit_tot_pixel[c][r] -> SetParLimits( par, pllo[par], plhi[par]);
            hist_tot_pixel[c][r]->Fit(Form("langau_c%d_r%d", c, r), "RQ");
            fit_tot_pixel[c][r]->GetParameters(fp);

            float tMPV = fp[1];
            float chisqr = fit_tot_pixel[c][r]->GetChisquare();
            float ndf = fit_tot_pixel[c][r]->GetNDF();

            hist_MPV -> Fill( tMPV );
            //file << c << ',' << r << ',' << hit << ',' << tMPV << ',' << chisqr/ndf << std::endl;

            MPV[c][r] = tMPV;



        }
    }

    //file.close();
    std::cout << "CSV file written successfully." << std::endl;

    // Create canvas and divide it into nc*nr pads
    TCanvas* cToT = new TCanvas("ToT", "ToT Dist.", 200 * nc, 200 * nr);
    cToT->Divide(nc, nr, 0.0005, 0.0005);

    TPad* pToT[nc*nr];

    int index = 0;
    for (int row: rows_to_draw) {
        for (int col : cols_to_draw) {
            printf("%f, ", MPV[col][row]);
            cToT -> cd(index+1);
            index++;

            gStyle->SetOptStat(11);
            gStyle->SetOptFit(111);

            /*
            pToT[index] = new TPad( Form("Pad_%d", index+1), Form("%d", index+1), 0, 0, 1, 1); 
            pToT[index] -> SetMargin(.13, .05, .12, .05);
            //pToT[index] -> SetLogy();
            pToT[index] -> Draw();
            pToT[index] -> cd();

            TH1* FrameToT = pToT[index] -> DrawFrame( 0, 0, MAXTOT, 250 );
            FrameToT -> GetXaxis() -> SetLabelSize(0.05);
            FrameToT -> GetXaxis() -> SetTitleSize(0.05);
            //FrameToT -> GetXaxis() -> SetTitle( Form("PMT%d ADC_{integral} [ns #times ADC]", b+1) );
            FrameToT -> GetYaxis() -> SetLabelSize(0.05);
            FrameToT -> GetYaxis() -> SetTitleSize(0.05);

            FrameToT -> Draw();
            */

            hist_tot_pixel[col][row]->Draw("SAME");
            if(MPV[col][row]<0.1) continue;
            fit_tot_pixel[col][row]->Draw("L SAME");

            TLatex* t1 = new TLatex();
            t1 -> SetTextSize( 0.08 );
            t1 -> DrawLatexNDC(.5, .5, Form("MPV = %.2f", MPV[col][row]) );
        }
        printf("\n");
    }
    /*
    TCanvas* c2 = new TCanvas("c2", "MPV Dist.", 800, 800);
    c2 -> cd();
    hist_MPV -> Draw();
    TFitResultPtr fit_result = hist_MPV->Fit("gaus", "SQ");
    */

    std::string figdir = "./fig";
    TString Figpath;
    Figpath.Form("%s/%s_c%dr%d.pdf",figdir.c_str(), file_name.c_str(), cols_to_draw[0], rows_to_draw[nr-1] );
    //cToT->SaveAs((figdir + "/" + file_name + 'c'+ cols_to_draw[0] + 'r' + rows_to_draw[0] + ".pdf").c_str());
    cToT -> SaveAs( Figpath );
    return;



}

// Function to load a .txt file and return a vector of values
std::vector<double> load_txt_file(const std::string& directory, int col, int row) {
    std::string filename = "ToT_distribution_col" + std::to_string(col) + "_row" + std::to_string(row) + ".txt";
    std::string filepath = directory + "/" + filename;

    std::ifstream file(filepath);
    std::vector<double> data;

    if (!file.is_open()) {
        std::cerr << "File " << filepath << " not found." << std::endl;
        return data;  // Return empty vector if file not found
    }

    double value;
    while (file >> value) {
        data.push_back(value);  // Read values from the file
    }

    return data;
}

double langaufun(double *x, double *par) {
 
   //Fit parameters:
   //par[0]=Width (scale) parameter of Landau density
   //par[1]=Most Probable (MP, location) parameter of Landau density
   //par[2]=Total area (integral -inf to inf, normalization constant)
   //par[3]=Width (sigma) of convoluted Gaussian function
   //
   //In the Landau distribution (represented by the CERNLIB approximation),
   //the maximum is located at x=-0.22278298 with the location parameter=0.
   //This shift is corrected within this function, so that the actual
   //maximum is identical to the MP parameter.
 
      // Numeric constants
      double invsq2pi = 0.3989422804014;   // (2 pi)^(-1/2)
      double mpshift  = -0.22278298;       // Landau maximum location
 
      // Control constants
      double np = 100.0;      // number of convolution steps
      double sc =   5.0;      // convolution extends to +-sc Gaussian sigmas
 
      // Variables
      double xx;
      double mpc;
      double fland;
      double sum = 0.0;
      double xlow,xupp;
      double step;
      double i;
 
 
      // MP shift correction
      mpc = par[1] - mpshift * par[0];
 
      // Range of convolution integral
      xlow = x[0] - sc * par[3];
      xupp = x[0] + sc * par[3];
 
      step = (xupp-xlow) / np;
 
      // Convolution integral of Landau and Gaussian by sum
      for(i=1.0; i<=np/2; i++) {
         xx = xlow + (i-.5) * step;
         fland = TMath::Landau(xx,mpc,par[0]) / par[0];
         sum += fland * TMath::Gaus(x[0],xx,par[3]);
 
         xx = xupp - (i-.5) * step;
         fland = TMath::Landau(xx,mpc,par[0]) / par[0];
         sum += fland * TMath::Gaus(x[0],xx,par[3]);
      }
 
      return (par[2] * step * sum * invsq2pi / par[3]);
}
 