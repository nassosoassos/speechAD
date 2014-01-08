function proto = create_proto_model(name, n_states, streams,feature_type)
% Create model prototype given state and stream information

proto.name = name;
proto.prior = zeros(1,n_states);
proto.prior(1) = 1;
proto.exiting_probs = zeros(1,n_states);
proto.exiting_probs(end) = 0.5;

n_streams = length(streams.weights);
proto.vector_type = feature_type;
for k=1:n_streams
    proto.streams(k).mu = zeros(streams.vec_sizes(k),n_states);
    proto.streams(k).sweights = streams.weights(k)*ones(1,n_states);
    proto.streams(k).Sigma = ones(streams.vec_sizes(k),n_states);
end

proto.transmat = zeros(n_states, n_states);
proto.transmat(1:n_states,1:n_states) = 0.5*eye(n_states,n_states)+diag(0.5*ones(1,n_states-1),1);