

(define (problem BW-rand-3)
(:domain blocksworld-4ops)
(:objects a b c )
(:init
(handempty)
(on a c)
(ontable b)
(on c b)
(clear a)
)
(:goal
(and
(on a b)
(on b c))
)
)


