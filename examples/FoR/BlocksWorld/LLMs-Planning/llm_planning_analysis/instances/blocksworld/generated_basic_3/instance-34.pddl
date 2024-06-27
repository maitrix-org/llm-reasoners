

(define (problem BW-rand-3)
(:domain blocksworld-4ops)
(:objects a b c )
(:init
(handempty)
(ontable a)
(ontable b)
(on c b)
(clear a)
(clear c)
)
(:goal
(and
(on b c))
)
)


