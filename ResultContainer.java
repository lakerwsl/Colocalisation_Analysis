import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import mpicbg.imglib.image.Image;
import mpicbg.imglib.type.numeric.RealType;

/**
 * The ResultContainer keeps all the results generated by the various
 * colocalisation algorithms. It allows a client to iterate over its
 * contents and makes the source image and channel information available
 * to a client.

 * @param <T>
 */
public class ResultContainer<T extends RealType<T>> implements Iterable<Result> {

	// The source images that the results are based on
	Image<T> sourceImage1, sourceImage2;
	// The channels of the source images that the result relate to
	int ch1, ch2;
	// The container of the results
	List<Result> resultsObjectList = new ArrayList<Result>();

	/**
	 * Adds a {@link Result} to the container.
	 *
	 * @param result The result to add.
	 */
	public void add(Result result){
		resultsObjectList.add(result);
	}

	/**
	 * Gets an iterator over the contained results.
	 */
	public Iterator<Result> iterator() {
		return resultsObjectList.iterator();
	}
}