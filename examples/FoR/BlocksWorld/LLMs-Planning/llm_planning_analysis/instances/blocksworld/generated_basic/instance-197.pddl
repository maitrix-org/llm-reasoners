

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(on b d)
(on c b)
(ontable d)
(clear a)
(clear c)
)
(:goal
(and
(on a d)
(on c a))
)
)


