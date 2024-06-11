

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a c)
(on b d)
(on c b)
(ontable d)
(clear a)
)
(:goal
(and
(on a d)
(on b c)
(on d b))
)
)


