import { gql } from '@apollo/client';
import * as Apollo from '@apollo/client';
export type Maybe<T> = T | null;
export type InputMaybe<T> = Maybe<T>;
export type Exact<T extends { [key: string]: unknown }> = { [K in keyof T]: T[K] };
export type MakeOptional<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]?: Maybe<T[SubKey]> };
export type MakeMaybe<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]: Maybe<T[SubKey]> };
const defaultOptions = {} as const;
/** All built-in and custom scalars, mapped to their actual values */
export type Scalars = {
  ID: string;
  String: string;
  Boolean: boolean;
  Int: number;
  Float: number;
  Upload: any;
};

export type AnimalInput = {
  image: Scalars['Upload'];
};

export type AnimalPredictionResponse = {
  __typename?: 'AnimalPredictionResponse';
  error?: Maybe<Error>;
  ok: Scalars['Boolean'];
  prediction?: Maybe<Prediction>;
};

export type Error = {
  __typename?: 'Error';
  field: Scalars['String'];
  message: Scalars['String'];
};

export type MetaResponse = {
  __typename?: 'MetaResponse';
  description: Scalars['String'];
  language: Scalars['String'];
  libraries: Array<Scalars['String']>;
  main: Scalars['String'];
  programmer: Scalars['String'];
};

export type Mutation = {
  __typename?: 'Mutation';
  predictAnimal: AnimalPredictionResponse;
};


export type MutationPredictAnimalArgs = {
  input: AnimalInput;
};

export type Predicted = {
  __typename?: 'Predicted';
  className: Scalars['String'];
  label: Scalars['Int'];
  probability: Scalars['Float'];
};

export type Prediction = {
  __typename?: 'Prediction';
  predictions: Array<Predicted>;
  topPrediction: Predicted;
};

export type Query = {
  __typename?: 'Query';
  meta: MetaResponse;
};

export type ErrorFragmentFragment = { __typename?: 'Error', field: string, message: string };

export type PredictedFragmentFragment = { __typename?: 'Predicted', label: number, probability: number, className: string };

export type PredictionFragmentFragment = { __typename?: 'Prediction', predictions: Array<{ __typename?: 'Predicted', label: number, probability: number, className: string }>, topPrediction: { __typename?: 'Predicted', label: number, probability: number, className: string } };

export type ClassifyAnimalMutationVariables = Exact<{
  input: AnimalInput;
}>;


export type ClassifyAnimalMutation = { __typename?: 'Mutation', predictAnimal: { __typename?: 'AnimalPredictionResponse', ok: boolean, error?: { __typename?: 'Error', field: string, message: string } | null, prediction?: { __typename?: 'Prediction', predictions: Array<{ __typename?: 'Predicted', label: number, probability: number, className: string }>, topPrediction: { __typename?: 'Predicted', label: number, probability: number, className: string } } | null } };

export const ErrorFragmentFragmentDoc = gql`
    fragment ErrorFragment on Error {
  field
  message
}
    `;
export const PredictedFragmentFragmentDoc = gql`
    fragment PredictedFragment on Predicted {
  label
  probability
  className
}
    `;
export const PredictionFragmentFragmentDoc = gql`
    fragment PredictionFragment on Prediction {
  predictions {
    ...PredictedFragment
  }
  topPrediction {
    ...PredictedFragment
  }
}
    ${PredictedFragmentFragmentDoc}`;
export const ClassifyAnimalDocument = gql`
    mutation ClassifyAnimal($input: AnimalInput!) {
  predictAnimal(input: $input) {
    ok
    error {
      ...ErrorFragment
    }
    prediction {
      ...PredictionFragment
    }
  }
}
    ${ErrorFragmentFragmentDoc}
${PredictionFragmentFragmentDoc}`;
export type ClassifyAnimalMutationFn = Apollo.MutationFunction<ClassifyAnimalMutation, ClassifyAnimalMutationVariables>;

/**
 * __useClassifyAnimalMutation__
 *
 * To run a mutation, you first call `useClassifyAnimalMutation` within a React component and pass it any options that fit your needs.
 * When your component renders, `useClassifyAnimalMutation` returns a tuple that includes:
 * - A mutate function that you can call at any time to execute the mutation
 * - An object with fields that represent the current status of the mutation's execution
 *
 * @param baseOptions options that will be passed into the mutation, supported options are listed on: https://www.apollographql.com/docs/react/api/react-hooks/#options-2;
 *
 * @example
 * const [classifyAnimalMutation, { data, loading, error }] = useClassifyAnimalMutation({
 *   variables: {
 *      input: // value for 'input'
 *   },
 * });
 */
export function useClassifyAnimalMutation(baseOptions?: Apollo.MutationHookOptions<ClassifyAnimalMutation, ClassifyAnimalMutationVariables>) {
        const options = {...defaultOptions, ...baseOptions}
        return Apollo.useMutation<ClassifyAnimalMutation, ClassifyAnimalMutationVariables>(ClassifyAnimalDocument, options);
      }
export type ClassifyAnimalMutationHookResult = ReturnType<typeof useClassifyAnimalMutation>;
export type ClassifyAnimalMutationResult = Apollo.MutationResult<ClassifyAnimalMutation>;
export type ClassifyAnimalMutationOptions = Apollo.BaseMutationOptions<ClassifyAnimalMutation, ClassifyAnimalMutationVariables>;

      export interface PossibleTypesResultData {
        possibleTypes: {
          [key: string]: string[]
        }
      }
      const result: PossibleTypesResultData = {
  "possibleTypes": {}
};
      export default result;
    