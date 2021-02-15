/*
 * Copyright (c) 2021 Anthony J. Greenberg
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/// Metropolis sampler
/** \file
 * \author Anthony J. Greenberg
 * \copyright Copyright (c) 2021 Anthony J. Greenberg
 * \version 1.0
 *
 * Class implementation for a simple Metropolis sampler with a Gaussian proposal.
 *
 */

#include "metropolis.hpp"

using namespace BayesicSpace;

// SamplerMetro methods
SamplerMetro::SamplerMetro(SamplerMetro &&in){
	if (&in != this) {
		model_    = in.model_;
		theta_    = in.theta_;
		incr_     = in.incr_;
		in.model_ = nullptr;
		in.theta_ = nullptr;
	}
}

SamplerMetro& SamplerMetro::operator=(SamplerMetro &&in){
	if (&in != this) {
		model_    = in.model_;
		theta_    = in.theta_;
		incr_     = in.incr_;
		in.model_ = nullptr;
		in.theta_ = nullptr;
	}
	return *this;
}

int16_t SamplerMetro::adapt(){
	vector<double> thetaPrime = *theta_;
	for (auto &t : thetaPrime) {
		t += incr_ * rng_.rnorm();
	}
	double lAlpha = model_->logPost(thetaPrime) - model_->logPost(*theta_);
	double lU     = log( rng_.runifnz() );
	if (lU < lAlpha) {
		(*theta_) = move(thetaPrime);
		return 1;
	} else {
		return 0;
	}
}

int16_t SamplerMetro::update(){
	vector<double> thetaPrime = *theta_;
	for (auto &t : thetaPrime) {
		t += incr_ * rng_.rnorm();
	}
	double lAlpha = model_->logPost(thetaPrime) - model_->logPost(*theta_);
	double lU     = log( rng_.runifnz() );
	if (lU < lAlpha) {
		(*theta_) = move(thetaPrime);
		return 1;
	} else {
		return 0;
	}
}
