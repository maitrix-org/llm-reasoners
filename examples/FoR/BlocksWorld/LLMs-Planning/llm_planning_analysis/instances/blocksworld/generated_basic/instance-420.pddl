

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(on b a)
(on c b)
(on d c)
(clear d)
)
(:goal
(and
(on a c)
(on d b))
)
)


