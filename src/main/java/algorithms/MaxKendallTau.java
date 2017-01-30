package algorithms;

import gadgets.DataContainer;
import ij.IJ;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.List;
import java.util.Collections;

import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.PairIterator;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.TwinCursor;
import net.imglib2.img.Img;
import net.imglib2.type.logic.BitType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;
import results.ResultHandler;

public class MaxKendallTau <T extends RealType< T >> extends Algorithm<T> {
	
	int nrRandomizations;
	
	public double maxtau;
	public double pvalue;
	public int thresholdRank1;
	public int thresholdRank2;
	public double[] sampleDistribution;
	
	public MaxKendallTau() {
		super("Maximum of Kendall's Tau Rank Correlation");
		this.nrRandomizations = 1000;
	}
	
	@Override
	public void execute(DataContainer<T> container)
		throws MissingPreconditionException
	{
		RandomAccessible<T> img1 = container.getSourceImage1();
		RandomAccessible<T> img2 = container.getSourceImage2();
		RandomAccessibleInterval<BitType> mask = container.getMask();

		TwinCursor<T> cursor = new TwinCursor<T>(img1.randomAccess(),
				img2.randomAccess(), Views.iterable(mask).localizingCursor());		
		
		maxtau = calculateMaxTauIndex(cursor);
		
		List<IterableInterval<T>> blockIntervals = generateBlocks(img1, container);
		Img<T> shuffledImage;
		sampleDistribution = new double[nrRandomizations];
		
		for (int i=0; i < nrRandomizations; i++) {
			shuffledImage = shuffleBlocks(blockIntervals, img1);
			cursor = new TwinCursor<T>(shuffledImage.randomAccess(),
					img2.randomAccess(), Views.iterable(mask).localizingCursor());
			sampleDistribution[i] = calculateMaxTauIndex(cursor);
		}
		
		pvalue = calculatePvalue(maxtau, sampleDistribution);
		
	}
	
	protected double calculatePvalue (double value, double[] distribution) {
		double count = 0;
		for (int i = 0; i <  distribution.length; i++) {
			if (distribution[i] > value)
			{
				count++;
			}
		}
		
		double pvalue = count / distribution.length;
		
		return pvalue;
	}
	
	protected <T extends RealType<T>> Img<T> shuffleBlocks(List<IterableInterval<T>> blockIntervals, RandomAccessible<T> img) {
		List<Cursor<T>> inputBlocks = new ArrayList<Cursor<T>>(nrBlocksPerImage);
		List<Cursor<T>> outputBlocks = new ArrayList<Cursor<T>>(nrBlocksPerImage);
		for (IterableInterval<T> roiIt : blockIntervals) {
			inputBlocks.add(roiIt.localizingCursor());
			outputBlocks.add(roiIt.localizingCursor());
		}
		
		final T zero = img.randomAccess().get().createVariable();
		zero.setZero();

		final long[] dims = new long[img.numDimensions()];
		img1.dimensions(dims);
		ImgFactory<T> factory = new ArrayImgFactory<T>();
		Img<T> shuffledImage = factory.create(dims, img.randomAccess().get().createVariable());
		RandomAccessible<T> infiniteShuffledImage = Views.extendValue(shuffledImage, zero );
		
		Collections.shuffle( inputBlocks );
		RandomAccess<T> output = infiniteShuffledImage.randomAccess();
		
		for (int j = 0; j < inputBlocks.size(); j++) {
			Cursor<T> inputCursor = inputBlocks.get(j);
			Cursor<T> outputCursor = outputBlocks.get(j);
			while (inputCursor.hasNext() && outputCursor.hasNext()) {
				inputCursor.fwd();
				outputCursor.fwd();
				output.setPosition(outputCursor);
				output.get().set( inputCursor.get() );
				inputCursor.reset();
				outputCursor.reset();
			}
		}
		
		return shuffledImage;
	}
	
	protected <T extends RealType<T>> List<IterableInterval<T>> generateBlocks(RandomAccessible<T> img, DataContainer<T> container) {
		long[] dimensions = container.getMaskBBSize();
		int nrDimensions = dimensions.length;
		int nrBlocksPerImage = 1;
		long[] nrBlocksPerDimension = new long[nrDimensions];
		long[] blockSize = new long[nrDimensions];
		
		for (int i = 0; i < nrDimensions; i++) {
			blockSize[i] = (long) Math.floor(Math.sqrt(dimensions[i]));
			nrBlocksPerDimension[i] = (long) (dimensions[i] / blockSize[i]);
			// if there is the need for a out-of-bounds block, increase count
			if ( dimensions[i] % blockSize[i] != 0 )
				nrBlocksPerDimension[i]++;
			nrBlocksPerImage *= nrBlocksPerDimension[i];
		}
		
		double[] floatOffset = new double[ img.numDimensions() ];
		long[] longOffset = container.getMaskBBOffset();
		for (int i=0; i< longOffset.length; ++i )
			floatOffset[i] = longOffset[i];
		double[] floatDimensions = new double[ nrDimensions ];
		for (int i=0; i< nrDimensions; ++i )
			floatDimensions[i] = dimensions[i];
		
		List<IterableInterval<T>> blockIntervals;
		blockIntervals = new ArrayList<IterableInterval<T>>( nrBlocksPerImage );
		RandomAccessible< T> infiniteImg = Views.extendMirrorSingle( img );
		generateBlocks( infiniteImg, blockIntervals, floatOffset, floatDimensions);
		
		return blockIntervals;
		
	}
	
	protected void generateBlocks(RandomAccessible<T> img, List<IterableInterval<T>> blockList,
			double[] offset, double[] size)
			throws MissingPreconditionException {
		// get the number of dimensions
		int nrDimensions = img.numDimensions();
		if (nrDimensions == 2)
		{ // for a 2D image...
			generateBlocksXY(img, blockList, offset, size);
		}
		else if (nrDimensions == 3)
		{ // for a 3D image...
			final double depth = size[2];
			double z;
			double originalZ = offset[2];
			// go through the depth in steps of block depth
			for ( z = psfRadius[2]; z <= depth; z += psfRadius[2] ) {

				offset[2] = originalZ + z - psfRadius[2];
				generateBlocksXY(img, blockList, offset, size);
			}
			// check is we need to add a out of bounds strategy cursor
			if (z > depth) {
				offset[2] = originalZ + z - psfRadius[2];
				generateBlocksXY(img, blockList, offset, size);
			}
			offset[2] = originalZ;
		}
		else
			throw new MissingPreconditionException("Currently only 2D and 3D images are supported.");
	}
	
	protected void generateBlocksXY(RandomAccessible<T> img, List<IterableInterval<T>> blockList,
			double[] offset, double[] size) {
		// potentially masked image height
		double height = size[1];
		final double originalY = offset[1];
		// go through the height in steps of block width
		double y;
		for ( y = psfRadius[1]; y <= height; y += psfRadius[1] ) {
			offset[1] = originalY + y - psfRadius[1];
			generateBlocksX(img, blockList, offset, size);
		}
		// check is we need to add a out of bounds strategy cursor
		if (y > height) {
			offset[1] = originalY + y - psfRadius[1];
			generateBlocksX(img, blockList, offset, size);
		}
		offset[1] = originalY;
	}
	
	protected void generateBlocksX(RandomAccessible<T> img, List<IterableInterval<T>> blockList,
			double[] offset, double[] size) {
		// potentially masked image width
		double width = size[0];
		final double originalX = offset[0];
		// go through the width in steps of block width
		double x;
		for ( x = psfRadius[0]; x <= width; x += psfRadius[0] ) {
			offset[0] = originalX + x - psfRadius[0];
			RectangleRegionOfInterest roi =
					new RectangleRegionOfInterest(offset.clone(), psfRadius.clone());
			IterableInterval<T> roiInterval = roi.getIterableIntervalOverROI(img);
			blockList.add(roiInterval);
		}
		// check is we need to add a out of bounds strategy cursor
		if (x > width) {
			offset[0] = originalX + x - psfRadius[0];
			RectangleRegionOfInterest roi =
					new RectangleRegionOfInterest(offset.clone(), psfRadius.clone());
			IterableInterval<T> roiInterval = roi.getIterableIntervalOverROI(img);
			blockList.add(roiInterval);
		}
		offset[0] = originalX;
	}
	
	protected <T extends RealType<T>> double calculateMaxTauIndex(final PairIterator<T> iterator) {
		double[][] values;
		double[][] rank;
		double maxtau;
		
		int capacity = 0;
		while (iterator.hasNext()) {
			iterator.fwd();
			capacity++;
		}
		iterator.reset();
		
		values = dataPreprocessing(iterator, capacity);
		
		double[] values1 = new double[capacity];
		double[] values2 = new double[capacity];
		for (int i = 0; i < capacity; i++)
		{
			values1[i] = values[i][0];
			values2[i] = values[i][1];
		}
		
		thresholdRank1 = calculateOtsuThreshold(values1);
		thresholdRank2 = calculateOtsuThreshold(values2);
		rank = rankTransformation(values, thresholdRank1, thresholdRank2, capacity);
		
		maxtau = calculateMaxKendallTau(rank, thresholdRank1, thresholdRank2, capacity);
		
		return maxtau;
		
	}
	
	protected <T extends RealType<T>> double[][] dataPreprocessing(final PairIterator<T> iterator, int capacity) {
		double[][] values = new double[capacity][2];
		iterator.reset();
		int count = 0;
		while (iterator.hasNext()) {
			iterator.fwd();
			values[count][0] = iterator.getFirst().getRealDouble();
			values[count][1] = iterator.getSecond().getRealDouble();
			count++;
		}
		
		return values;
	}
	
	protected double calculateOtsuThreshold(double[] data) {		
		double sortdata = data.clone();
		Arrays.sort(sortdata);
		
		int start = 0;
		int end = 0;
		int bestThre;
		double bestVar = -1;
		int L = sortdata.length;
		double lessIns = 0;
		int lessNum = 0;
		double largeIns;
		int largeNum;
		double lessMean;
		double largeMean;
		double diffMean;
		double varBetween;
		int tempNum = 0;
		double totalSum = 0;
		for (int i = 0; i < L; i++)
			totalSum += sortdata[i];
		
		while (end < L - 1)
		{
			while (Double.compare(sortdata[start],sortdata[end]) == 0)
				end++;	
			tempNum = end-start;
			lessNum += tempNum;
			largeNum = L - lessNum;
			lessIns += tempNum*sortdata[end-1];
			largeIns = totalSum - lessIns;
			lessMean = lessIns / lessNum;
			largeMean = largeIns / largeNum;
			diffMean = largeMean - lessMean;
			varBetween = lessNum * largeNum * diffMean * diffMean;
			
			if (varBetween > bestVar)
			{
				bestVar = varBetween;
				bestThre = largeNum;
			}
			start = end; 
		}
		
		if (bestThre < L/2)
			bestThre = L/2;
		
		return bestThre;
	}
	
	protected double[][] rankTransformation(final double[][] values, double thresholdRank1, double thresholdRank2, int n) {	
		double[][] tempRank = new double[n][2];
		for( int i = 0; i < n; i++) {
			tempRank[i][0] = values[i][0];
			tempRank[i][1] = values[i][1];
		}
		
		Arrays.sort(tempRank, new Comparator<double[]>() {
			@Override
			public int compare(double[] row1, double[] row2) {
				return Double.compare(row1[1], row2[1]);
			}
		});
		
		int start = 0;
		int end = 0;
		int rank=0;
		while (end < n-1)
		{
			while (Double.compare(tempRank[start][1],tempRank[end][1]) == 0)
				end++;
			for (int i = start; i < end; i++){
				tempRank[i][1]=rank+Math.random();
			}
			rank++;
			start=end;
		}
		
		Arrays.sort(tempRank, new Comparator<double[]>() {
			@Override
			public int compare(double[] row1, double[] row2) {
				return Double.compare(row1[1], row2[1]);
			}
		});
		
		for (int i = 0; i < n; i++) {
			tempRank[i][1] = i + 1;
		}
		
		
		//second 
		Arrays.sort(tempRank, new Comparator<double[]>() {
			@Override
			public int compare(double[] row1, double[] row2) {
				return Double.compare(row1[0], row2[0]);
			}
		});
		
		start = 0;
		end = 0;
		rank=0;
		while (end < n-1)
		{
			while (Double.compare(tempRank[start][0],tempRank[end][0]) == 0)
				end++;
			for (int i = start; i < end; i++){
				tempRank[i][0]=rank+Math.random();
			}
			rank++;
			start=end;
		}
		
		Arrays.sort(tempRank, new Comparator<double[]>() {
			@Override
			public int compare(double[] row1, double[] row2) {
				return Double.compare(row1[0], row2[0]);
			}
		});
		
		for (int i = 0; i < n; i++) {
			tempRank[i][0] = i + 1;
		}
		
		
		List<Integer> validIndex = new ArrayList<Integer>();
		for (int i = 0; i < n; i++)
		{
			if(tempRank[i][0] >= thresholdRank1 && tempRank[i][1] >= thresholdRank2)
			{
				validIndex.add(i);
			}
		}
		
		int rn=validIndex.size();
		double[][] rank = new double[rn][2];
		int index = 0;
		for( Integer i : validIndex ) {
			rank[index][0] = tempRank[i][0];
			rank[index][1] = tempRank[i][1];
			index++;
		}
		
		return rank;
	}
	
	protected double calculateMaxKendallTau(final double[][] rank, double thresholdRank1, double thresholdRank2, int n) {
		int rn = rank.length;
		int an;
		double step = 1+1.0/Math.log(Math.log(n));
		double tempOff1=1;
		double tempOff2;
		List<Integer> activeIndex;
		double sdTau;
		double kendallTau;
		double normalTau;
		double maxNormalTau = Double.MIN_VALUE;
		
		while (tempOff1*step+thresholdRank1<n) {
			tempOff1 *= step;
			tempOff2 = 1;
			while (tempOff2*step+thresholdRank2<n) {
				tempOff2 *= step;
				
				activeIndex = new ArrayList<Integer>();
				for (int i = 0; i < rn; i++)
				{
					if(rank[i][0] >= n - tempOff1 && rank[i][1] >= n - tempOff2)
					{
						activeIndex.add(i);
					}
				}
				an = activeIndex.size();
				if (an > 1)
				{
					kendallTau = calculateKendallTau(rank, activeIndex);
					sdTau = Math.sqrt(2 * (2 * an + 5) / 9 / an / (an - 1));
					normalTau = kendallTau / sdTau;
				}
				else
				{
					normalTau = Double.MIN_VALUE
				}
				if (normalTau > maxNormalTau)
					maxNormalTau = normalTau;
			}
		}
		
		return maxNormalTau;
	}
	
	protected double calculateKendallTau(final double[][] rank, List<Integer> activeIndex) {
		int an = activeIndex.size();
		double[][] partRank = new double[an][2];
		int index = 0;
		for( Integer i : activeIndex ) {
			partRank[index][0] = rank[i][0];
			partRank[index][1] = rank[i][1];
			index++;
		}
		
		int[] index = new int[an];
		for (int i = 0; i < an; i++) {
			index[i] = i;
		}
		
		IntArraySorter.sort(index, new IntComparator() {
			@Override
			public int compare(int a, int b) {
				double xa = partRank[a][0];
				double xb = partRank[b][0];
				return Double.compare(xa, xb);
			}
		});
		
		final MergeSort mergeSort = new MergeSort(index, new IntComparator() {

			@Override
			public int compare(int a, int b) {
				double ya = partRank[a][1];
				double yb = partRank[b][1];
				return Double.compare(ya, yb);
			}
		});
		
		long n0 = an * (long)(an - 1) / 2;
		long S = mergeSort.sort();
		
		return (n0 - 2 * S) / (double)n0;
		
	}
	
	private final static class MergeSort {

		private int[] index;
		private final IntComparator comparator;

		public MergeSort(int[] index, IntComparator comparator) {
			this.index = index;
			this.comparator = comparator;
		}

		public int[] getSorted() {
			return index;
		}

		/**
		 * Sorts the {@link #index} array.
		 * <p>
		 * This implements a non-recursive merge sort.
		 * </p>
		 * @param begin
		 * @param end
		 * @return the equivalent number of BubbleSort swaps
		 */
		public long sort() {
			long swaps = 0;
			int n = index.length;
			// There are merge sorts which perform in-place, but their runtime is worse than O(n log n)
			int[] index2 = new int[n];
			for (int step = 1; step < n; step <<= 1) {
				int begin = 0, k = 0;
				for (;;) {
					int begin2 = begin + step, end = begin2 + step;
					if (end >= n) {
						if (begin2 >= n) {
							break;
						}
						end = n;
					}

					// calculate the equivalent number of BubbleSort swaps
					// and perform merge, too
					int i = begin, j = begin2;
					while (i < begin2 && j < end) {
						int compare = comparator.compare(index[i], index[j]);
						if (compare > 0) {
							swaps += (begin2 - i);
							index2[k++] = index[j++];
						} else {
							index2[k++] = index[i++];
						}
					}
					if (i < begin2) {
						do {
							index2[k++] = index[i++];
						} while (i < begin2);
					} else {
						while (j < end) {
							index2[k++] = index[j++];
						}
					}
					begin = end;
				}
				if (k < n) {
					System.arraycopy(index, k, index2, k, n - k);
				}
				int[] swapIndex = index2;
				index2 = index;
				index = swapIndex;
			}

			return swaps;
		}

	}
	
	@Override
	public void processResults(ResultHandler<T> handler) {
		super.processResults(handler);
		handler.handleValue("Max Kendall Tau correlation value", maxtau, 4);
		handler.handleValue("P-value", pvalue, 4);
	}

}