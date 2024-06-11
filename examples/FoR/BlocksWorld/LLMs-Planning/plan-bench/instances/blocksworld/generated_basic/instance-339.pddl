

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a c)
(ontable b)
(on c b)
(on d a)
(clear d)
)
(:goal
(and
(on b a)
(on c d)
(on d b))
)
)


