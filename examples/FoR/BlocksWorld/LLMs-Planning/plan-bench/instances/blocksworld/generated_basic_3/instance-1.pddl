

(define (problem BW-rand-3)
(:domain blocksworld-4ops)
(:objects a b c )
(:init
(handempty)
(ontable a)
(on b a)
(on c b)
(clear c)
)
(:goal
(and
(on a b)
(on c a))
)
)


